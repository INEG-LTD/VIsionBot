"""
WhileGoal implementation for loop-like goal execution.

Runs a sub-goal repeatedly until a condition is achieved, similar to:
  - while not CONDITION: do ACTION
  - ACTION until CONDITION

Now supports two distinct evaluation routes:
- "see": Vision-based condition evaluation using screenshots
- "page": Page-based condition evaluation using DOM/state detection

This composes the existing Conditional/Condition infra and integrates with
the retry mechanism so the planner keeps iterating the body until done.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, Union

from .base import BaseGoal, GoalResult, GoalStatus, EvaluationTiming, Condition, GoalContext, create_environment_condition


# Type alias for JSON expressions
JsonExpr = Union[Dict[str, Any], list, str, int, float, bool, None]

@dataclass
class LoopProgress:
    iterations: int = 0
    last_condition_result: Optional[bool] = None
    started_at: float = 0.0
    loop_retries: int = 0  # Track custom loop retries separately from goal retries


class WhileGoal(BaseGoal):
    """
    Goal that keeps requesting retries (i.e., new plans) until the condition
    evaluates to True. The provided `loop_goal` describes the body (what to do
    on each iteration), and its description is surfaced to planning prompts.
    
    Now supports two distinct evaluation routes:
    - "see": Vision-based condition evaluation using screenshots
    - "page": Page-based condition evaluation using DOM/state detection
    """

    # Evaluate after interactions to decide whether to continue
    EVALUATION_TIMING: EvaluationTiming = EvaluationTiming.AFTER

    def __init__(
        self,
        condition_text: str,  # Natural language condition
        loop_prompt: str,
        route: str,  # REQUIRED: "see" or "page"
        else_prompt: Optional[str] = None,
        description: Optional[str] = None,
        max_iterations: int = 30,
        **kwargs,
    ) -> None:
        if route not in ["see", "page"]:
            raise ValueError(f"Invalid route '{route}'. Must be 'see' or 'page'")
        
        # Align detection need with the body by default
        super().__init__(description, needs_detection=False, needs_plan=False, **kwargs)

        self.condition_text = condition_text
        self.loop_prompt = loop_prompt
        self.route = route
        self.else_prompt = else_prompt
        self.progress = LoopProgress(iterations=0, last_condition_result=None, started_at=time.time(), loop_retries=0)
        self.max_iterations = max(1, int(max_iterations))
        
        # Create condition with caching
        self.condition = self._create_condition()
        self._condition_cache: Dict[str, Any] = {}  # Cache for converted conditions
        

    def set_goal_monitor(self, goal_monitor) -> None:
        """Set the goal monitor."""
        self.goal_monitor = goal_monitor

    def _create_condition(self) -> Condition:
        """Create condition based on specified route"""
        if self.route == "see":
            return self._create_vision_condition()
        elif self.route == "page":
            return self._create_page_condition()
        else:
            raise ValueError(f"Invalid route: {self.route}")

    def _create_vision_condition(self) -> Condition:
        """Create vision-based condition using screenshots and natural language"""
        def evaluator(context: GoalContext) -> bool:
            try:
                page = context.page_reference
                if not page:
                    return False
                
                # Take screenshot
                screenshot = page.screenshot(type="jpeg", quality=50, full_page=False)
                
                # Convert condition to vision question (with caching)
                question = self._convert_to_vision_question(self.condition_text)
                
                # Use vision model to answer
                from ai_utils import answer_question_with_vision
                result = answer_question_with_vision(question, screenshot)
                
                # Parse result
                answer = str(result or "").lower().strip()
                return answer in ['yes', 'true', '1']
                
            except Exception as e:
                print(f"[WhileGoal] Vision condition error: {e}")
                return False
        
        return create_environment_condition(
            f"Vision condition: {self.condition_text}",
            evaluator
        )

    def _create_page_condition(self) -> Condition:
        """Create page-based condition using condition engine"""
        def evaluator(context: GoalContext) -> bool:
            try:
                # Convert natural language to JSON DSL (with caching)
                expr = self._convert_to_page_expression(self.condition_text)
                if not expr:
                    print(f"[WhileGoal] Failed to convert condition to page expression: {self.condition_text}")
                    return False
                
                # Use condition engine to evaluate
                from goals.condition_engine import get_default_engine
                engine = get_default_engine()
                outcome = engine.evaluate(expr, context)
                return outcome.value
                
            except Exception as e:
                print(f"[WhileGoal] Page condition error: {e}")
                return False
        
        return create_environment_condition(
            f"Page condition: {self.condition_text}",
            evaluator
        )

    def _convert_to_vision_question(self, condition_text: str) -> str:
        """Convert natural language condition to vision question (with caching)"""
        cache_key = f"vision_question::{condition_text}"
        if cache_key in self._condition_cache:
            return self._condition_cache[cache_key]
        
        from ai_utils import generate_text
        
        prompt = f"""
        Convert this condition into a yes/no question that can be answered by looking at a screenshot:
        
        Condition: {condition_text}
        
        Return only the question, no explanation.
        """
        
        result = generate_text(
            prompt=prompt,
            system_prompt="Convert conditions to yes/no questions for vision analysis.",
            reasoning_level="low"
        )
        
        question = str(result or condition_text).strip()
        self._condition_cache[cache_key] = question
        return question

    def _convert_to_page_expression(self, condition_text: str) -> Optional[JsonExpr]:
        """Convert natural language condition to JSON DSL expression (with caching)"""
        cache_key = f"page_expression::{condition_text}"
        if cache_key in self._condition_cache:
            return self._condition_cache[cache_key]
        
        from goals.condition_engine import compile_nl_to_expr
        
        expr = compile_nl_to_expr(condition_text)
        if expr:
            self._condition_cache[cache_key] = expr
        return expr

    def get_description(self, context: GoalContext) -> str:
        """Generate description with route information"""
        route_info = f"[{self.route.upper()}]"
        if self.else_prompt:
            return f"{route_info} While ({self.condition_text}) do: {self.loop_prompt} else: {self.else_prompt}"
        else:
            return f"{route_info} While ({self.condition_text}) do: {self.loop_prompt}"

    def evaluate(self, context: GoalContext) -> GoalResult:
        """Evaluate loop condition based on specified route"""
        try:
            cond = bool(self.condition.evaluator(context))
        except Exception as e:
            print(f"[WhileGoal] Condition evaluation error: {e}")
            cond = False
            
        self.progress.last_condition_result = cond

        # If condition is false -> loop terminates
        if not cond:
            if self.else_prompt:
                print(f"🔄 Executing else goal: {self.else_prompt}")
                return GoalResult(
                    status=GoalStatus.ACHIEVED,
                    confidence=1.0,
                    reasoning=f"Loop condition false, executing else goal: {self.else_prompt}",
                    evidence={"iterations": self.progress.iterations, "route": self.route},
                    next_actions=[f"Execute else: {self.else_prompt}"]
                )
            else:
                return GoalResult(
                    status=GoalStatus.ACHIEVED,
                    confidence=1.0,
                    reasoning=f"Loop condition false after {self.progress.iterations} iterations",
                    evidence={"iterations": self.progress.iterations, "route": self.route},
                    next_actions=[]
                )

        # Condition is true -> continue loop
        self.progress.iterations += 1
        
        if self.progress.iterations > self.max_iterations:
            return GoalResult(
                status=GoalStatus.FAILED,
                confidence=0.8,
                reasoning=f"Loop exceeded max iterations ({self.max_iterations})",
                evidence={"iterations": self.progress.iterations, "route": self.route},
                next_actions=[]
            )

        return GoalResult(
            status=GoalStatus.PENDING,
            confidence=0.9,
            reasoning=f"Loop condition true (iteration {self.progress.iterations}), executing loop body",
            evidence={"iterations": self.progress.iterations, "condition_result": cond, "route": self.route},
            next_actions=[f"Execute: {self.loop_prompt}"],
        )
