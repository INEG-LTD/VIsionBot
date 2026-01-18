"""
Simple, robust event-driven logging system for Browser.

Design principles:
- Non-blocking: logging errors never break the bot
- Simple: minimal API surface
- Flexible: easy to customize output via callbacks
"""
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import time
from utils.debug_print import dprint, PrintMode


class EventType(str, Enum):
    """All event types that can be logged"""
    # Agent events
    AGENT_START = "agent_start"
    AGENT_ITERATION = "agent_iteration"
    AGENT_COMPLETE = "agent_complete"
    AGENT_ERROR = "agent_error"
    
    # Action events
    ACTION_START = "action_start"
    ACTION_SUCCESS = "action_success"
    ACTION_FAILURE = "action_failure"
    ACTION_STEP = "action_step"  # Detailed step execution
    ACTION_COORDINATES = "action_coordinates"  # Coordinate selection
    ACTION_REFINEMENT = "action_refinement"  # Element refinement
    
    # Command events
    COMMAND_START = "command_start"
    COMMAND_SUCCESS = "command_success"
    COMMAND_FAILURE = "command_failure"
    
    # System events
    SYSTEM_INFO = "system_info"
    SYSTEM_WARNING = "system_warning"
    SYSTEM_ERROR = "system_error"
    SYSTEM_DEBUG = "system_debug"
    
    # Extraction events
    EXTRACTION_START = "extraction_start"
    EXTRACTION_SUCCESS = "extraction_success"
    EXTRACTION_FAILURE = "extraction_failure"
    
    # Planning events
    PLAN_GENERATED = "plan_generated"
    PLAN_CACHED = "plan_cached"
    PLAN_CLEARED = "plan_cleared"
    
    # Navigation events
    TAB_SWITCH = "tab_switch"
    TAB_NEW = "tab_new"
    
    # Completion events
    COMPLETION_CHECK = "completion_check"
    COMPLETION_SUCCESS = "completion_success"
    
    # Action determination events
    ACTION_DETERMINED = "action_determined"
    ACTION_PARAMS = "action_params"
    
    # Sub-agent events
    SUB_AGENT_POLICY = "sub_agent_policy"
    
    # Extraction events (already defined above, but adding detail events)
    EXTRACTION_DETECTED = "extraction_detected"
    
    # Tab events (already defined, but adding registration)
    TAB_REGISTERED = "tab_registered"
    
    # Performance/cost events
    LLM_COST = "llm_cost"  # Token usage and cost tracking
    
    # Command execution events
    COMMAND_EXECUTION_START = "command_execution_start"
    COMMAND_EXECUTION_COMPLETE = "command_execution_complete"
    OVERLAY_SELECTION = "overlay_selection"
    
    # Planning events (detailed)
    PLAN_OVERLAY_CANDIDATES = "plan_overlay_candidates"
    PLAN_OVERLAY_CHOSEN = "plan_overlay_chosen"
    
    # Command history
    COMMAND_HISTORY = "command_history"
    OVERLAY_DATA = "overlay_data"
    
    # Interaction tracking
    INTERACTION_RECORDED = "interaction_recorded"
    
    # Action state details
    ACTION_STATE_CHANGE = "action_state_change"

    # Task orchestration events
    TASK_DECOMPOSE_START = "task_decompose_start"
    TASK_DECOMPOSE_COMPLETE = "task_decompose_complete"
    TASK_DECOMPOSE_FAIL = "task_decompose_fail"
    TASK_START = "task_start"
    TASK_COMPLETE = "task_complete"
    TASK_FAIL = "task_fail"

    # Sequential task events
    SEQUENTIAL_START = "sequential_start"
    SEQUENTIAL_COMPLETE = "sequential_complete"
    SEQUENTIAL_FAIL = "sequential_fail"
    SEQUENTIAL_ITERATION_START = "sequential_iteration_start"
    SEQUENTIAL_ITERATION_COMPLETE = "sequential_iteration_complete"
    SEQUENTIAL_ITERATION_FAIL = "sequential_iteration_fail"
    SUBTASK_START = "subtask_start"
    SUBTASK_COMPLETE = "subtask_complete"
    SUBTASK_FAIL = "subtask_fail"

    # Mini-loop events
    MINILOOP_ITERATION_START = "miniloop_iteration_start"
    MINILOOP_ITERATION_COMPLETE = "miniloop_iteration_complete"
    MINILOOP_ITERATION_FAIL = "miniloop_iteration_fail"

    # Plan execution events
    PLAN_EXECUTE_START = "plan_execute_start"
    PLAN_EXECUTE_COMPLETE = "plan_execute_complete"
    PLAN_EXECUTE_FAIL = "plan_execute_fail"

    # Schema events
    SCHEMA_INFER_START = "schema_infer_start"
    SCHEMA_INFER_SUCCESS = "schema_infer_success"
    SCHEMA_INFER_FAIL = "schema_infer_fail"
    SCHEMA_VALIDATE_FAIL = "schema_validate_fail"

    # Extraction retries/empties
    EXTRACTION_RETRY = "extraction_retry"
    EXTRACTION_EMPTY = "extraction_empty"

    # Sequence planner events
    SEQUENCE_DECISION = "sequence_decision"
    SEQUENCE_RETRY = "sequence_retry"
    SEQUENCE_END = "sequence_end"

    # Tab lifecycle events
    TAB_CLOSE = "tab_close"
    TAB_DETECTED = "tab_detected"

    # Agent lifecycle events
    AGENT_PAUSE = "agent_pause"
    AGENT_RESUME = "agent_resume"

    # Retry/backoff events
    RETRY_BACKOFF = "retry_backoff"
    RETRY_GIVEUP = "retry_giveup"

    # Model selection events
    MODEL_FALLBACK = "model_fallback"

    # Middleware events
    MIDDLEWARE_BEFORE = "middleware_before"
    MIDDLEWARE_AFTER = "middleware_after"
    MIDDLEWARE_ERROR = "middleware_error"

    # Cache events
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    CACHE_STORE = "cache_store"
    CACHE_EVICT = "cache_evict"
    CACHE_CLEAR = "cache_clear"

    # Cost events
    COST_WARNING = "cost_warning"
    COST_LIMIT_EXCEEDED = "cost_limit_exceeded"

    # Human-in-loop events
    HUMAN_PAUSE = "human_pause"
    HUMAN_RESUME = "human_resume"

    # Context guard events
    CONTEXT_GUARD_START = "context_guard_start"
    CONTEXT_GUARD_DECISION = "context_guard_decision"
    CONTEXT_GUARD_CACHE = "context_guard_cache"

    # Action queue events
    QUEUE_ENQUEUE = "queue_enqueue"
    QUEUE_DEQUEUE = "queue_dequeue"
    QUEUE_CLEAR = "queue_clear"
    QUEUE_REJECT = "queue_reject"


@dataclass
class BotEvent:
    """Structured event data"""
    event_type: EventType
    message: str
    timestamp: float = field(default_factory=time.time)
    level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, SUCCESS
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "event_type": self.event_type.value,
            "message": self.message,
            "timestamp": self.timestamp,
            "timestamp_iso": datetime.fromtimestamp(self.timestamp).isoformat(),
            "level": self.level,
            "details": self.details
        }


class EventLogger:
    """
    Simple, robust event logger.
    
    In debug mode: prints directly to console
    In normal mode: only calls callbacks (no prints)
    """
    
    def __init__(self, debug_mode: bool = True, show_overlay_candidates: bool = False):
        try:
            self.debug_mode = debug_mode
            self.show_overlay_candidates = show_overlay_candidates
            self._callbacks: List[Callable[[BotEvent], None]] = []
            self._event_history: List[BotEvent] = []
            self._max_history = 1000
        except Exception:
            # If even initialization fails, set minimal defaults
            self.debug_mode = True
            self.show_overlay_candidates = False
            self._callbacks = []
            self._event_history = []
            self._max_history = 1000
        
    def register_callback(self, callback: Callable[[BotEvent], None]) -> None:
        """Register a callback for all events"""
        if callback not in self._callbacks:
            self._callbacks.append(callback)
    
    def _safe_emit(self, event: BotEvent) -> None:
        """Safely emit an event - never raises exceptions"""
        # Store in history
        try:
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history.pop(0)
        except Exception:
            pass  # Ignore history errors
        
        # Debug mode: print directly
        if self.debug_mode:
            try:
                self._print_event(event)
            except Exception:
                pass  # Ignore print errors
        
        # Call callbacks (normal mode or in addition to debug prints)
        for callback in self._callbacks:
            try:
                callback(event)
            except Exception:
                pass  # Ignore callback errors
    
    def _print_event(self, event: BotEvent) -> None:
        """Print event in debug mode"""
        level_emoji = {
            "DEBUG": "🔍",
            "INFO": "ℹ️",
            "WARNING": "⚠️",
            "ERROR": "❌",
            "SUCCESS": "✅"
        }
        emoji = level_emoji.get(event.level, "•")
        dprint(f"{emoji} {event.message}")
        
        # Print important details
        if event.details:
            for key, value in event.details.items():
                if value is not None and key not in ['timestamp', 'timestamp_iso']:
                    # Only print simple types to avoid errors
                    try:
                        if isinstance(value, (str, int, float, bool)):
                            dprint(f"   {key}: {value}")
                    except Exception:
                        pass
    
    def emit(self, event_type: EventType, message: str, level: str = "INFO", **details) -> None:
        """Emit an event - safe wrapper that never raises"""
        try:
            event = BotEvent(
                event_type=event_type,
                message=message,
                level=level,
                details=details
            )
            self._safe_emit(event)
        except Exception:
            # Last resort: if even creating the event fails, try to print in debug mode
            if self.debug_mode:
                try:
                    dprint(f"⚠️ Event logger error: {message}")
                except Exception:
                    pass
    
    # Convenience methods - all wrapped in try/except for safety
    def agent_start(self, prompt: str, **details):
        try:
            self.emit(EventType.AGENT_START, f"Starting agentic mode: {prompt}", "INFO", prompt=prompt, **details)
        except Exception:
            pass
    
    def agent_iteration(self, iteration: int, max_iterations: int, url: str = None, title: str = None, **details):
        try:
            msg = f"Iteration {iteration}/{max_iterations}"
            if url:
                msg += f" - {url}"
            self.emit(EventType.AGENT_ITERATION, msg, "INFO", 
                     iteration=iteration, max_iterations=max_iterations, url=url, title=title, **details)
        except Exception:
            pass
    
    def agent_complete(self, success: bool, reasoning: str = None, confidence: float = None, **details):
        try:
            status = "completed successfully" if success else "failed"
            level = "SUCCESS" if success else "ERROR"
            msg = f"Agent {status}"
            if reasoning:
                msg += f": {reasoning}"
            self.emit(EventType.AGENT_COMPLETE, msg, level, 
                     success=success, reasoning=reasoning, confidence=confidence, **details)
        except Exception:
            pass

    def agent_error(self, message: str, **details):
        try:
            self.emit(EventType.AGENT_ERROR, f"Agent error: {message}", "ERROR", message=message, **details)
        except Exception:
            pass
    
    def command_start(self, command: str, command_id: str = None, **details):
        try:
            msg = f"Starting command: {command}"
            if command_id:
                msg += f" [ID: {command_id}]"
            self.emit(EventType.COMMAND_START, msg, "INFO",
                     command=command, command_id=command_id, **details)
        except Exception:
            pass

    def command_success(self, command: str, **details):
        try:
            msg = f"Command completed: {command}"
            self.emit(EventType.COMMAND_SUCCESS, msg, "SUCCESS", command=command, **details)
        except Exception:
            pass

    def command_failure(self, command: str, error: str = None, **details):
        try:
            msg = f"Command failed: {command}"
            if error:
                msg += f" - {error}"
            self.emit(EventType.COMMAND_FAILURE, msg, "ERROR", command=command, error=error, **details)
        except Exception:
            pass
    
    def system_info(self, message: str, **details):
        try:
            self.emit(EventType.SYSTEM_INFO, message, "INFO", **details)
        except Exception:
            pass
    
    def system_warning(self, message: str, **details):
        try:
            self.emit(EventType.SYSTEM_WARNING, message, "WARNING", **details)
        except Exception:
            pass
    
    def system_error(self, message: str, error: Exception = None, **details):
        try:
            msg = message
            if error:
                msg += f" - {str(error)}"
            self.emit(EventType.SYSTEM_ERROR, msg, "ERROR", error=str(error) if error else None, **details)
        except Exception:
            pass
    
    def system_debug(self, message: str, **details):
        try:
            self.emit(EventType.SYSTEM_DEBUG, message, "DEBUG", **details)
        except Exception:
            pass
    
    def extraction_start(self, prompt: str, **details):
        try:
            self.emit(EventType.EXTRACTION_START, f"Extracting: {prompt}", "INFO", prompt=prompt, **details)
        except Exception:
            pass
    
    def extraction_success(self, prompt: str, result: Any = None, **details):
        try:
            self.emit(EventType.EXTRACTION_SUCCESS, f"Extraction completed: {prompt}", "SUCCESS", prompt=prompt, result=str(result) if result else None, **details)
        except Exception:
            pass
    
    def extraction_failure(self, prompt: str, error: str = None, **details):
        try:
            msg = f"Extraction failed: {prompt}"
            if error:
                msg += f" - {error}"
            self.emit(EventType.EXTRACTION_FAILURE, msg, "ERROR", prompt=prompt, error=error, **details)
        except Exception:
            pass
    
    def plan_cached(self, command: str, **details):
        try:
            self.emit(EventType.PLAN_CACHED, f"Cached plan for command '{command}'", "INFO",
                     command=command, **details)
        except Exception:
            pass
    
    def plan_cleared(self, reason: str, **details):
        try:
            self.emit(EventType.PLAN_CLEARED, f"Clearing cached plan ({reason})", "INFO", reason=reason, **details)
        except Exception:
            pass

    def overlay_data_snapshot(self, count: int, preview: str = None, **details):
        try:
            msg = f"Overlay data captured: {count} elements"
            if preview:
                msg += f" ({preview})"
            self.emit(EventType.OVERLAY_DATA, msg, "DEBUG", count=count, preview=preview, **details)
        except Exception:
            pass

    def plan_generated(self, plan_reasoning: str, confidence: float, expected_outcome: str, steps_summary: str = "", **details):
        try:
            msg = f"Plan generated (confidence {confidence:.2f})"
            if plan_reasoning:
                msg += f": {plan_reasoning}"
            if expected_outcome:
                msg += f" → Expected: {expected_outcome}"
            self.emit(
                EventType.PLAN_GENERATED,
                msg,
                "INFO",
                plan_reasoning=plan_reasoning,
                confidence=confidence,
                expected_outcome=expected_outcome,
                steps_summary=steps_summary,
                **details,
            )
        except Exception:
            pass
    
    def plan_reused(self, **details):
        try:
            self.emit(EventType.PLAN_CACHED, "Reusing cached plan (skipped LLM planning)", "INFO", **details)
        except Exception:
            pass
    
    def tab_switch(self, tab_id: str, url: str = None, **details):
        try:
            msg = f"Switched to tab: {tab_id}"
            if url:
                msg += f" ({url})"
            self.emit(EventType.TAB_SWITCH, msg, "INFO", tab_id=tab_id, url=url, **details)
        except Exception:
            pass
    
    def tab_new(self, tab_id: str, url: str = None, **details):
        try:
            msg = f"New tab registered: {tab_id}"
            if url:
                msg += f" ({url})"
            self.emit(EventType.TAB_NEW, msg, "INFO", tab_id=tab_id, url=url, **details)
        except Exception:
            pass
    
    def completion_check(self, is_complete: bool, reasoning: str = None, confidence: float = None, **details):
        try:
            status = "complete" if is_complete else "not complete"
            msg = f"Task {status}"
            if reasoning:
                msg += f": {reasoning}"
            level = "SUCCESS" if is_complete else "INFO"
            event_type = EventType.COMPLETION_SUCCESS if is_complete else EventType.COMPLETION_CHECK
            self.emit(event_type, msg, level, is_complete=is_complete, reasoning=reasoning, confidence=confidence, **details)
        except Exception:
            pass

    def agent_completed(self, reasoning: str = None, **details):
        """Log when agent signals task completion via 'complete:' command"""
        try:
            msg = "✅ Agent signaled task completion"
            if reasoning:
                msg += f"\n   Reasoning: {reasoning}"
            self.emit(EventType.COMPLETION_SUCCESS, msg, "SUCCESS", completion_type="agent_signaled", reasoning=reasoning, **details)
        except Exception:
            pass

    def action_determined(self, action: str, reasoning: str = None, confidence: float = None, expected_outcome: str = None, **details):
        try:
            msg = f"Next action determined: {action}"
            if reasoning:
                msg += f"\n   Reasoning: {reasoning}"
            if confidence is not None:
                msg += f"\n   Confidence: {confidence:.2f}"
            if expected_outcome:
                msg += f"\n   Expected outcome: {expected_outcome}"
            self.emit(EventType.ACTION_DETERMINED, msg, "INFO", action=action, reasoning=reasoning, confidence=confidence, expected_outcome=expected_outcome, **details)
        except Exception:
            pass
    
    def action_params(self, params: dict, **details):
        try:
            msg = "act() parameters:"
            for key, value in params.items():
                msg += f"\n   {key}: {value}"
            self.emit(EventType.ACTION_PARAMS, msg, "DEBUG", **params, **details)
        except Exception:
            pass
    
    def sub_agent_policy(self, policy: str, score: float = None, reason: str = None, **details):
        try:
            msg = f"Sub-agent utilization policy → {policy}"
            if score is not None:
                msg += f" (score {score:.2f})"
            if reason:
                msg += f"\n   Reason: {reason}"
            self.emit(EventType.SUB_AGENT_POLICY, msg, "INFO", policy=policy, score=score, reason=reason, **details)
        except Exception:
            pass
    
    def tab_registered(self, tab_id: str, purpose: str = None, url: str = None, **details):
        try:
            msg = f"Registered tab: {tab_id}"
            if purpose:
                msg += f" ({purpose})"
            if url:
                msg += f" - {url}"
            self.emit(EventType.TAB_REGISTERED, msg, "INFO", tab_id=tab_id, purpose=purpose, url=url, **details)
        except Exception:
            pass
    
    def extraction_detected(self, prompt: str, **details):
        try:
            self.emit(EventType.EXTRACTION_DETECTED, f"Extraction detected: {prompt}", "INFO", prompt=prompt, **details)
        except Exception:
            pass
    
    def action_step(self, step_number: int, action_type: str, **details):
        try:
            self.emit(EventType.ACTION_STEP, f"Step {step_number}: {action_type}", "DEBUG", step_number=step_number, action_type=action_type, **details)
        except Exception:
            pass

    def action_start(self, action_type: str, **details):
        try:
            self.emit(EventType.ACTION_START, f"Action start: {action_type}", "INFO", action_type=action_type, **details)
        except Exception:
            pass

    def action_success(self, action_type: str, **details):
        try:
            self.emit(EventType.ACTION_SUCCESS, f"Action success: {action_type}", "SUCCESS", action_type=action_type, **details)
        except Exception:
            pass

    def action_failure(self, action_type: str, error: str = None, **details):
        try:
            msg = f"Action failed: {action_type}"
            if error:
                msg += f" - {error}"
            self.emit(EventType.ACTION_FAILURE, msg, "ERROR", action_type=action_type, error=error, **details)
        except Exception:
            pass

    def task_decompose_start(self, prompt: str, **details):
        try:
            self.emit(EventType.TASK_DECOMPOSE_START, "Task decomposition started", "INFO", prompt=prompt, **details)
        except Exception:
            pass

    def task_decompose_complete(self, task_count: int, **details):
        try:
            self.emit(EventType.TASK_DECOMPOSE_COMPLETE, f"Task decomposition complete: {task_count} tasks", "SUCCESS", task_count=task_count, **details)
        except Exception:
            pass

    def task_decompose_fail(self, error: str, **details):
        try:
            self.emit(EventType.TASK_DECOMPOSE_FAIL, f"Task decomposition failed: {error}", "ERROR", error=error, **details)
        except Exception:
            pass

    def task_start(self, task_id: str, description: str, **details):
        try:
            self.emit(EventType.TASK_START, f"Task start: {description}", "INFO", task_id=task_id, description=description, **details)
        except Exception:
            pass

    def task_complete(self, task_id: str, description: str, **details):
        try:
            self.emit(EventType.TASK_COMPLETE, f"Task complete: {description}", "SUCCESS", task_id=task_id, description=description, **details)
        except Exception:
            pass

    def task_fail(self, task_id: str, description: str, error: str = None, **details):
        try:
            msg = f"Task failed: {description}"
            if error:
                msg += f" - {error}"
            self.emit(EventType.TASK_FAIL, msg, "ERROR", task_id=task_id, description=description, error=error, **details)
        except Exception:
            pass

    def sequential_start(self, task_id: str, goal: str, **details):
        try:
            self.emit(EventType.SEQUENTIAL_START, f"Sequential task start: {goal}", "INFO", task_id=task_id, goal=goal, **details)
        except Exception:
            pass

    def sequential_complete(self, task_id: str, goal: str, **details):
        try:
            self.emit(EventType.SEQUENTIAL_COMPLETE, f"Sequential task complete: {goal}", "SUCCESS", task_id=task_id, goal=goal, **details)
        except Exception:
            pass

    def sequential_fail(self, task_id: str, goal: str, error: str = None, **details):
        try:
            msg = f"Sequential task failed: {goal}"
            if error:
                msg += f" - {error}"
            self.emit(EventType.SEQUENTIAL_FAIL, msg, "ERROR", task_id=task_id, goal=goal, error=error, **details)
        except Exception:
            pass

    def sequential_iteration_start(self, task_id: str, iteration: int, **details):
        try:
            self.emit(EventType.SEQUENTIAL_ITERATION_START, f"Sequential iteration start: {iteration}", "INFO", task_id=task_id, iteration=iteration, **details)
        except Exception:
            pass

    def sequential_iteration_complete(self, task_id: str, iteration: int, **details):
        try:
            self.emit(EventType.SEQUENTIAL_ITERATION_COMPLETE, f"Sequential iteration complete: {iteration}", "SUCCESS", task_id=task_id, iteration=iteration, **details)
        except Exception:
            pass

    def sequential_iteration_fail(self, task_id: str, iteration: int, error: str = None, **details):
        try:
            msg = f"Sequential iteration failed: {iteration}"
            if error:
                msg += f" - {error}"
            self.emit(EventType.SEQUENTIAL_ITERATION_FAIL, msg, "ERROR", task_id=task_id, iteration=iteration, error=error, **details)
        except Exception:
            pass

    def subtask_start(self, instruction: str, **details):
        try:
            self.emit(EventType.SUBTASK_START, f"Subtask start: {instruction}", "INFO", instruction=instruction, **details)
        except Exception:
            pass

    def subtask_complete(self, instruction: str, **details):
        try:
            self.emit(EventType.SUBTASK_COMPLETE, f"Subtask complete: {instruction}", "SUCCESS", instruction=instruction, **details)
        except Exception:
            pass

    def subtask_fail(self, instruction: str, error: str = None, **details):
        try:
            msg = f"Subtask failed: {instruction}"
            if error:
                msg += f" - {error}"
            self.emit(EventType.SUBTASK_FAIL, msg, "ERROR", instruction=instruction, error=error, **details)
        except Exception:
            pass

    def miniloop_iteration_start(self, task_instruction: str, iteration: int, **details):
        try:
            self.emit(EventType.MINILOOP_ITERATION_START, f"Mini-loop iteration start: {iteration}", "DEBUG", task_instruction=task_instruction, iteration=iteration, **details)
        except Exception:
            pass

    def miniloop_iteration_complete(self, task_instruction: str, iteration: int, **details):
        try:
            self.emit(EventType.MINILOOP_ITERATION_COMPLETE, f"Mini-loop iteration complete: {iteration}", "DEBUG", task_instruction=task_instruction, iteration=iteration, **details)
        except Exception:
            pass

    def miniloop_iteration_fail(self, task_instruction: str, iteration: int, error: str = None, **details):
        try:
            msg = f"Mini-loop iteration failed: {iteration}"
            if error:
                msg += f" - {error}"
            self.emit(EventType.MINILOOP_ITERATION_FAIL, msg, "WARNING", task_instruction=task_instruction, iteration=iteration, error=error, **details)
        except Exception:
            pass

    def plan_execute_start(self, step_count: int, **details):
        try:
            self.emit(EventType.PLAN_EXECUTE_START, f"Plan execution start ({step_count} steps)", "INFO", step_count=step_count, **details)
        except Exception:
            pass

    def plan_execute_complete(self, **details):
        try:
            self.emit(EventType.PLAN_EXECUTE_COMPLETE, "Plan execution completed", "SUCCESS", **details)
        except Exception:
            pass

    def plan_execute_fail(self, error: str = None, **details):
        try:
            msg = "Plan execution failed"
            if error:
                msg += f" - {error}"
            self.emit(EventType.PLAN_EXECUTE_FAIL, msg, "ERROR", error=error, **details)
        except Exception:
            pass

    def schema_infer_start(self, goal: str, **details):
        try:
            self.emit(EventType.SCHEMA_INFER_START, "Schema inference started", "DEBUG", goal=goal, **details)
        except Exception:
            pass

    def schema_infer_success(self, fields: list[str], **details):
        try:
            self.emit(EventType.SCHEMA_INFER_SUCCESS, f"Schema inferred: {fields}", "SUCCESS", fields=fields, **details)
        except Exception:
            pass

    def schema_infer_fail(self, error: str, **details):
        try:
            self.emit(EventType.SCHEMA_INFER_FAIL, f"Schema inference failed: {error}", "ERROR", error=error, **details)
        except Exception:
            pass

    def schema_validate_fail(self, error: str, **details):
        try:
            self.emit(EventType.SCHEMA_VALIDATE_FAIL, f"Schema validation failed: {error}", "WARNING", error=error, **details)
        except Exception:
            pass

    def extraction_retry(self, prompt: str, attempt: int, **details):
        try:
            self.emit(EventType.EXTRACTION_RETRY, f"Extraction retry {attempt}: {prompt}", "WARNING", prompt=prompt, attempt=attempt, **details)
        except Exception:
            pass

    def extraction_empty(self, prompt: str, **details):
        try:
            self.emit(EventType.EXTRACTION_EMPTY, f"Extraction empty: {prompt}", "WARNING", prompt=prompt, **details)
        except Exception:
            pass

    def sequence_decision(self, decision: str, **details):
        try:
            self.emit(EventType.SEQUENCE_DECISION, f"Sequence decision: {decision}", "INFO", decision=decision, **details)
        except Exception:
            pass

    def sequence_retry(self, iteration: int, **details):
        try:
            self.emit(EventType.SEQUENCE_RETRY, f"Sequence retry iteration {iteration}", "WARNING", iteration=iteration, **details)
        except Exception:
            pass

    def sequence_end(self, reason: str, **details):
        try:
            self.emit(EventType.SEQUENCE_END, f"Sequence end: {reason}", "INFO", reason=reason, **details)
        except Exception:
            pass

    def tab_close(self, tab_id: str, **details):
        try:
            self.emit(EventType.TAB_CLOSE, f"Tab closed: {tab_id}", "INFO", tab_id=tab_id, **details)
        except Exception:
            pass

    def tab_detected(self, tab_id: str, **details):
        try:
            self.emit(EventType.TAB_DETECTED, f"Tab detected: {tab_id}", "INFO", tab_id=tab_id, **details)
        except Exception:
            pass

    def agent_pause(self, message: str = None, **details):
        try:
            msg = "Agent paused"
            if message:
                msg += f": {message}"
            self.emit(EventType.AGENT_PAUSE, msg, "INFO", message=message, **details)
        except Exception:
            pass

    def agent_resume(self, **details):
        try:
            self.emit(EventType.AGENT_RESUME, "Agent resumed", "INFO", **details)
        except Exception:
            pass

    def retry_backoff(self, delay_seconds: float, attempt: int, **details):
        try:
            self.emit(EventType.RETRY_BACKOFF, f"Retry backoff {attempt}: {delay_seconds:.2f}s", "WARNING", delay_seconds=delay_seconds, attempt=attempt, **details)
        except Exception:
            pass

    def retry_giveup(self, reason: str = None, **details):
        try:
            msg = "Retry give up"
            if reason:
                msg += f": {reason}"
            self.emit(EventType.RETRY_GIVEUP, msg, "ERROR", reason=reason, **details)
        except Exception:
            pass

    def model_fallback(self, primary: str, fallback: str, **details):
        try:
            self.emit(EventType.MODEL_FALLBACK, f"Model fallback: {primary} -> {fallback}", "WARNING", primary=primary, fallback=fallback, **details)
        except Exception:
            pass

    def middleware_before(self, middleware_name: str, action_type: str = None, **details):
        try:
            msg = f"Middleware before: {middleware_name}"
            self.emit(EventType.MIDDLEWARE_BEFORE, msg, "DEBUG", middleware=middleware_name, action_type=action_type, **details)
        except Exception:
            pass

    def middleware_after(self, middleware_name: str, action_type: str = None, **details):
        try:
            msg = f"Middleware after: {middleware_name}"
            self.emit(EventType.MIDDLEWARE_AFTER, msg, "DEBUG", middleware=middleware_name, action_type=action_type, **details)
        except Exception:
            pass

    def middleware_error(self, middleware_name: str, action_type: str = None, error: str = None, **details):
        try:
            msg = f"Middleware error: {middleware_name}"
            self.emit(EventType.MIDDLEWARE_ERROR, msg, "ERROR", middleware=middleware_name, action_type=action_type, error=error, **details)
        except Exception:
            pass

    def cache_hit(self, cache_key: str = None, **details):
        try:
            self.emit(EventType.CACHE_HIT, "Cache hit", "DEBUG", cache_key=cache_key, **details)
        except Exception:
            pass

    def cache_miss(self, cache_key: str = None, **details):
        try:
            self.emit(EventType.CACHE_MISS, "Cache miss", "DEBUG", cache_key=cache_key, **details)
        except Exception:
            pass

    def cache_store(self, cache_key: str = None, **details):
        try:
            self.emit(EventType.CACHE_STORE, "Cache store", "DEBUG", cache_key=cache_key, **details)
        except Exception:
            pass

    def cache_evict(self, cache_key: str = None, **details):
        try:
            self.emit(EventType.CACHE_EVICT, "Cache evict", "WARNING", cache_key=cache_key, **details)
        except Exception:
            pass

    def cache_clear(self, **details):
        try:
            self.emit(EventType.CACHE_CLEAR, "Cache cleared", "INFO", **details)
        except Exception:
            pass

    def cost_warning(self, total_cost: float, max_cost: float, **details):
        try:
            msg = f"Cost warning: ${total_cost:.4f} / ${max_cost:.2f}"
            self.emit(EventType.COST_WARNING, msg, "WARNING", total_cost=total_cost, max_cost=max_cost, **details)
        except Exception:
            pass

    def cost_limit_exceeded(self, total_cost: float, max_cost: float, **details):
        try:
            msg = f"Cost limit exceeded: ${total_cost:.4f} > ${max_cost:.2f}"
            self.emit(EventType.COST_LIMIT_EXCEEDED, msg, "ERROR", total_cost=total_cost, max_cost=max_cost, **details)
        except Exception:
            pass

    def human_pause(self, message: str, **details):
        try:
            self.emit(EventType.HUMAN_PAUSE, f"Human pause: {message}", "INFO", message=message, **details)
        except Exception:
            pass

    def human_resume(self, **details):
        try:
            self.emit(EventType.HUMAN_RESUME, "Human resumed", "INFO", **details)
        except Exception:
            pass

    def context_guard_start(self, guard_text: str, overlay_index: int = None, **details):
        try:
            self.emit(EventType.CONTEXT_GUARD_START, "Context guard start", "DEBUG", guard_text=guard_text, overlay_index=overlay_index, **details)
        except Exception:
            pass

    def context_guard_decision(self, passed: bool, reason: str = None, cached: bool = False, **details):
        try:
            level = "SUCCESS" if passed else "WARNING"
            msg = "Context guard passed" if passed else "Context guard failed"
            self.emit(EventType.CONTEXT_GUARD_DECISION, msg, level, passed=passed, reason=reason, cached=cached, **details)
        except Exception:
            pass

    def context_guard_cache(self, guard_text: str, overlay_index: int = None, **details):
        try:
            self.emit(EventType.CONTEXT_GUARD_CACHE, "Context guard cache hit", "DEBUG", guard_text=guard_text, overlay_index=overlay_index, **details)
        except Exception:
            pass

    def queue_enqueue(self, action_id: str = None, **details):
        try:
            self.emit(EventType.QUEUE_ENQUEUE, "Queue enqueue", "DEBUG", action_id=action_id, **details)
        except Exception:
            pass

    def queue_dequeue(self, action_id: str = None, **details):
        try:
            self.emit(EventType.QUEUE_DEQUEUE, "Queue dequeue", "DEBUG", action_id=action_id, **details)
        except Exception:
            pass

    def queue_clear(self, **details):
        try:
            self.emit(EventType.QUEUE_CLEAR, "Queue cleared", "INFO", **details)
        except Exception:
            pass

    def queue_reject(self, reason: str, action_id: str = None, **details):
        try:
            msg = f"Queue reject: {reason}"
            self.emit(EventType.QUEUE_REJECT, msg, "WARNING", reason=reason, action_id=action_id, **details)
        except Exception:
            pass
    
    def action_coordinates(self, message: str, **details):
        try:
            self.emit(EventType.ACTION_COORDINATES, message, "DEBUG", **details)
        except Exception:
            pass
    
    def action_refinement(self, message: str, **details):
        try:
            self.emit(EventType.ACTION_REFINEMENT, message, "DEBUG", **details)
        except Exception:
            pass
    
    def llm_cost(self, cost_usd: float, input_tokens: int, output_tokens: int, total_tokens: int, model: str = None, **details):
        try:
            msg = f"Prompt Cost: {cost_usd} USD, Input Tokens: {input_tokens}, Output Tokens: {output_tokens}, Total Tokens: {total_tokens}"
            if model:
                msg += f" (Model: {model})"
            self.emit(EventType.LLM_COST, msg, "DEBUG", cost_usd=cost_usd, input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=total_tokens, model=model, **details)
        except Exception:
            pass
    
    def command_execution_start(self, instruction: str, target_hint: str = None, **details):
        try:
            msg = f"Executing command for instruction='{instruction}'"
            if target_hint:
                msg += f" target_hint='{target_hint}'"
            self.emit(EventType.COMMAND_EXECUTION_START, msg, "DEBUG", instruction=instruction, target_hint=target_hint, **details)
        except Exception:
            pass
    
    def command_execution_complete(self, command: str, success: bool = True, **details):
        try:
            msg = f"Command execution completed: {command}"
            level = "SUCCESS" if success else "ERROR"
            self.emit(EventType.COMMAND_EXECUTION_COMPLETE, msg, level, command=command, success=success, **details)
        except Exception:
            pass
    
    def overlay_selection(self, message: str, **details):
        try:
            self.emit(EventType.OVERLAY_SELECTION, message, "DEBUG", **details)
        except Exception:
            pass
    
    def plan_overlay_candidates(self, candidates: List[str], **details):
        try:
            if not self.show_overlay_candidates:
                return
            msg = "Candidate overlays for LLM selection:"
            for candidate in candidates:
                msg += f"\n  • {candidate}"
            self.emit(EventType.PLAN_OVERLAY_CANDIDATES, msg, "DEBUG", candidates=candidates, **details)
        except Exception:
            pass
    
    def plan_overlay_chosen(self, overlay_index: int, raw_response: str = None, **details):
        try:
            msg = f"overlay #{overlay_index} chosen"
            if raw_response:
                msg += f" (raw='{raw_response}')"
            self.emit(EventType.PLAN_OVERLAY_CHOSEN, msg, "DEBUG", overlay_index=overlay_index, raw_response=raw_response, **details)
        except Exception:
            pass
    
    def command_history(self, command: str, **details):
        try:
            self.emit(EventType.COMMAND_HISTORY, f"Added to command history: '{command}'", "DEBUG", command=command, **details)
        except Exception:
            pass
    
    def interaction_recorded(self, interaction_type: str, **details):
        try:
            message = f"Recorded {interaction_type} interaction"
            if details.get('reasoning'):
                message += f"\n   Why: {details['reasoning']}"
            self.emit(EventType.INTERACTION_RECORDED, message, "DEBUG", interaction_type=interaction_type, **details)
        except Exception:
            pass
    
    def action_state_change(self, message: str, url_before: str = None, url_after: str = None, dom_changed: bool = None, **details):
        try:
            self.emit(EventType.ACTION_STATE_CHANGE, message, "INFO", url_before=url_before, url_after=url_after, dom_changed=dom_changed, **details)
        except Exception:
            pass
    


# Global instance
_global_event_logger: Optional[EventLogger] = None

def get_event_logger() -> EventLogger:
    """Get the global event logger instance"""
    global _global_event_logger
    if _global_event_logger is None:
        _global_event_logger = EventLogger(debug_mode=True, show_overlay_candidates=False)  # Default to debug for backward compatibility
    return _global_event_logger

def set_event_logger(logger: EventLogger) -> None:
    """Set the global event logger instance"""
    global _global_event_logger
    _global_event_logger = logger
