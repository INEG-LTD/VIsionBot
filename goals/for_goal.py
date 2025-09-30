"""
ForGoal implementation with vision-based target resolution.

Runs a sub-goal repeatedly against resolved targets, similar to:
  - for each TARGET: do ACTION
  - for N times: do ACTION
  - for each ITEM in LIST: do ACTION

This uses a two-phase approach:
1. RESOLVE PHASE: Identify and classify iteration targets using vision
2. EXECUTE PHASE: Execute loop body against each resolved target
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Union

from .base import BaseGoal, GoalResult, GoalStatus, EvaluationTiming, GoalContext
from .target_resolver import TargetResolver
from .contextual_command_engine import ContextualCommandEngine


@dataclass
class ForLoopProgress:
    """Track progress through for loop execution"""
    iterations: int = 0
    target_iterations: int = 0
    started_at: float = 0.0
    loop_retries: int = 0
    current_target_index: int = 0
    targets_processed: List[str] = field(default_factory=list)
    successful_iterations: int = 0
    failed_iterations: int = 0
    skipped_iterations: int = 0
    phase: str = "RESOLVE"  # "RESOLVE" or "EXECUTE"


class ForGoal(BaseGoal):
    """
    For loop goal with vision-based target resolution.
    
    Supports multiple iteration modes:
    - count: Fixed number of iterations
    - elements: Iterate over page elements
    - items: Iterate over predefined list
    - pages: Iterate over pagination/navigation
    """
    
    EVALUATION_TIMING = EvaluationTiming.AFTER
    
    def __init__(
        self,
        iteration_mode: str,
        iteration_target: Union[int, List[str], str],
        loop_prompt: str,
        break_conditions: Optional[List[str]] = None,
        continue_conditions: Optional[List[str]] = None,
        skip_conditions: Optional[List[str]] = None,
        else_prompt: Optional[str] = None,
        description: Optional[str] = None,
        max_iterations: int = 100,
        **kwargs
    ) -> None:
        super().__init__(description, needs_detection=False, needs_plan=False, **kwargs)
        
        self.iteration_mode = iteration_mode
        self.iteration_target = iteration_target
        self.loop_prompt = loop_prompt
        self.break_conditions = break_conditions or []
        self.continue_conditions = continue_conditions or []
        self.skip_conditions = skip_conditions or []
        self.else_prompt = else_prompt
        self.max_iterations = max(1, int(max_iterations))
        
        # Progress tracking
        self.progress = ForLoopProgress(
            started_at=time.time(),
            target_iterations=self._calculate_target_iterations()
        )
        
        # Vision-based components
        self.target_resolver = TargetResolver()
        self.command_engine = ContextualCommandEngine()
        
        # Resolved targets (populated during RESOLVE phase)
        self.resolved_targets: List[Dict[str, Any]] = []
        self.current_target: Optional[Dict[str, Any]] = None
        
    def _calculate_target_iterations(self) -> int:
        """Calculate expected number of iterations based on mode and target"""
        if self.iteration_mode == "count":
            return int(self.iteration_target)
        elif self.iteration_mode == "elements":
            # Will be determined during resolution
            return 0
        elif self.iteration_mode == "items":
            return len(self.iteration_target) if isinstance(self.iteration_target, list) else 0
        elif self.iteration_mode == "pages":
            return 10  # Default pagination limit
        return 1
    
    def set_goal_monitor(self, goal_monitor) -> None:
        """Set the goal monitor and initialize vision components"""
        self.goal_monitor = goal_monitor
        
        # Initialize target resolver with page reference
        if hasattr(goal_monitor, 'page'):
            self.target_resolver.set_page(goal_monitor.page)
    
    def get_description(self, context: GoalContext) -> str:
        """Generate a detailed description of what this for loop is doing"""
        if self.progress.phase == "RESOLVE":
            return f"For loop: Resolving {self.iteration_mode} targets for '{self.iteration_target}'"
        elif self.progress.phase == "EXECUTE":
            if self.resolved_targets:
                remaining = len(self.resolved_targets) - self.progress.current_target_index
                return f"For loop: Executing against {len(self.resolved_targets)} targets ({remaining} remaining)"
            else:
                return f"For loop: No targets resolved"
        else:
            return f"For loop: {self.iteration_mode} iteration of '{self.iteration_target}'"
    
    def evaluate(self, context: GoalContext) -> GoalResult:
        """Evaluate for loop state and determine next action"""
        try:
            if self.progress.phase == "RESOLVE":
                return self._handle_resolve_phase(context)
            elif self.progress.phase == "EXECUTE":
                return self._handle_execute_phase(context)
            else:
                return GoalResult(
                    status=GoalStatus.FAILED,
                    confidence=0.0,
                    reasoning="Unknown for loop phase"
                )
        except Exception as e:
            return GoalResult(
                status=GoalStatus.FAILED,
                confidence=0.0,
                reasoning=f"For loop error: {str(e)}"
            )
    
    def _handle_resolve_phase(self, context: GoalContext) -> GoalResult:
        """Handle the target resolution phase"""
        print(f"🔍 RESOLVE PHASE: Identifying {self.iteration_mode} targets...")
        
        # Take screenshot for vision analysis
        screenshot = context.page_reference.screenshot()
        
        # Resolve targets based on iteration mode
        if self.iteration_mode == "count":
            # For count-based loops, create dummy targets
            self.resolved_targets = [
                {"index": i, "context": {"iteration": i + 1}}
                for i in range(int(self.iteration_target))
            ]
        elif self.iteration_mode == "elements":
            # Use vision to detect elements
            self.resolved_targets = self.target_resolver.resolve_element_targets(
                self.iteration_target, screenshot
            )
        elif self.iteration_mode == "items":
            # Convert items to targets
            self.resolved_targets = [
                {"index": i, "context": {"item": item}}
                for i, item in enumerate(self.iteration_target)
            ]
        else:
            return GoalResult(
                status=GoalStatus.FAILED,
                confidence=0.0,
                reasoning=f"Unsupported iteration mode: {self.iteration_mode}"
            )
        
        if not self.resolved_targets:
            return GoalResult(
                status=GoalStatus.FAILED,
                confidence=0.0,
                reasoning="No targets resolved for iteration"
            )
        
        print(f"🎯 Resolved {len(self.resolved_targets)} targets:")
        for i, target in enumerate(self.resolved_targets[:3]):  # Show first 3
            context_info = target.get('context', {})
            if 'job_title' in context_info:
                print(f"   • {context_info['job_title']} at {context_info.get('company', 'Unknown')}")
            elif 'item' in context_info:
                print(f"   • {context_info['item']}")
            else:
                print(f"   • Target {i + 1}")
        
        if len(self.resolved_targets) > 3:
            print(f"   ... and {len(self.resolved_targets) - 3} more targets")
        
        # Move to execute phase
        self.progress.phase = "EXECUTE"
        self.progress.target_iterations = len(self.resolved_targets)
        
        return GoalResult(
            status=GoalStatus.PENDING,
            confidence=0.8,
            reasoning=f"Resolved {len(self.resolved_targets)} targets, starting execution",
            next_actions=[self.loop_prompt]  # Execute the original command (ref: commands should not be rewritten)
        )
    
    def _handle_execute_phase(self, context: GoalContext) -> GoalResult:
        """Handle the execution phase"""
        if self.progress.current_target_index >= len(self.resolved_targets):
            # Loop completed
            print(f"🎉 For loop completed: {self.progress.successful_iterations}/{self.progress.target_iterations} iterations successful")
            
            return GoalResult(
                status=GoalStatus.ACHIEVED,
                confidence=1.0,
                reasoning=f"For loop completed successfully: {self.progress.successful_iterations}/{self.progress.target_iterations} iterations",
                evidence={
                    "iterations": self.progress.iterations,
                    "successful": self.progress.successful_iterations,
                    "failed": self.progress.failed_iterations,
                    "skipped": self.progress.skipped_iterations
                }
            )
        
        # Get current target
        self.current_target = self.resolved_targets[self.progress.current_target_index]
        
        print(f"🔄 Iteration {self.progress.current_target_index + 1}/{len(self.resolved_targets)}: Processing target")
        
        # Check if this is a ref command - don't rewrite ref commands
        if self.loop_prompt.strip().lower().startswith("ref:"):
            # For ref commands, execute the original command without rewriting
            contextual_command = self.loop_prompt
            print(f"   → Executing ref command: {contextual_command}")
        else:
            # For direct commands, rewrite for this target
            contextual_command = self.command_engine.rewrite_command(
                self.loop_prompt, 
                self.current_target,
                context.page_reference.screenshot()
            )
            print(f"   → Rewritten to: {contextual_command}")
        
        # Execute the contextual command
        # This will be handled by the main bot loop via retry mechanism
        return GoalResult(
            status=GoalStatus.PENDING,
            confidence=0.7,
            reasoning=f"Executing iteration {self.progress.current_target_index + 1} of {len(self.resolved_targets)}",
            next_actions=[contextual_command]
        )
    
    def on_iteration_complete(self, success: bool) -> None:
        """Called after each iteration completes"""
        self.progress.iterations += 1
        
        if success:
            self.progress.successful_iterations += 1
            print(f"   ✅ Iteration {self.progress.current_target_index + 1} successful")
        else:
            self.progress.failed_iterations += 1
            print(f"   ❌ Iteration {self.progress.current_target_index + 1} failed")
        
        # Move to next target
        self.progress.current_target_index += 1
        self.current_target = None
        
        # Check break conditions
        if self._should_break_loop():
            print("🛑 Break condition met, ending loop early")
            self.progress.current_target_index = len(self.resolved_targets)  # End loop
    
    def _should_break_loop(self) -> bool:
        """Check if any break conditions are met"""
        # This would be implemented with condition evaluation
        # For now, return False
        return False
    
    def request_retry(self, reason: str = "For loop iteration retry") -> bool:
        """Request retry for current iteration"""
        if self.progress.loop_retries >= 3:  # Max 3 retries per iteration
            print(f"⚠️ Max retries exceeded for iteration {self.progress.current_target_index + 1}")
            self.on_iteration_complete(False)
            return False
        
        self.progress.loop_retries += 1
        print(f"🔄 Retrying iteration {self.progress.current_target_index + 1} (attempt {self.progress.loop_retries + 1})")
        
        # Request retry from goal monitor
        if hasattr(self, 'goal_monitor') and hasattr(self.goal_monitor, 'request_retry'):
            return self.goal_monitor.request_retry(reason)
        
        return True
