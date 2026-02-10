"""
Simple, robust event-driven logging system for Agent.

Design principles:
- Non-blocking: logging errors never break the agent
- Simple: minimal API surface
- Flexible: easy to customize output via callbacks
"""
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import time
from models.models import ActionStep, NotebookEntryType, TaskDefinition
from utils.debug_print import dprint
import simplejson as json

class EventType(str, Enum):
    """All event types that can be logged"""
    # Agent events
    AGENT_START = "agent_start"
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
    
    ASK_REQUESTED = "ask_requested"
    
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
    ACTION_STEP_GENERATION_ERROR = "action_step_generation_error"
    
    # Completion events
    COMPLETION_CHECK = "completion_check"
    COMPLETION_SUCCESS = "completion_success"
    
    # Action determination events
    ACTION_DETERMINED = "action_determined"
    ACTION_ERROR = "action_error"
    ACTION_PLAN_GENERATION_ERROR = "action_plan_generation_error"
    
    # Extraction events (already defined above, but adding detail events)
    EXTRACTION_DETECTED = "extraction_detected"
    
    
    # Performance/cost events
    LLM_COST = "llm_cost"  # Token usage and cost tracking
    
    # Command execution events
    COMMAND_EXECUTION_START = "command_execution_start"
    COMMAND_EXECUTION_COMPLETE = "command_execution_complete"
    COMMAND_EXECUTION_FAILURE = "command_execution_failure"
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
    UNKNOWN_TASK_TYPE = "unknown_task_type"

    # Progress-based task events
    PROGRESS_MARKED = "progress_marked"
    TARGET_REVISED = "target_revised"
    STUCK_DETECTED = "stuck_detected"
    SUBTASK_START = "subtask_start"
    SUBTASK_COMPLETE = "subtask_complete"
    SUBTASK_FAIL = "subtask_fail"

    # Mini-loop events
    ITERATION_START = "iteration_start"
    ITERATION_COMPLETE = "iteration_complete"
    ITERATION_FAIL = "iteration_fail"

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



    # Agent lifecycle events
    AGENT_PAUSE = "agent_pause"
    AGENT_RESUME = "agent_resume"

    # Retry/backoff events
    RETRY_BACKOFF = "retry_backoff"
    RETRY_GIVEUP = "retry_giveup"

    # Model selection events
    MODEL_FALLBACK = "model_fallback"

    ASK_COMMAND_ANSWERED = "ask_command_answered"
    ASK_COMMAND_SKIPPED = "ask_command_skipped"
    ASK_COMMAND_FAILURE = "ask_command_failure"
    AGENT_TALK = "agent_talk"
    AGENT_TALK_FAILURE = "agent_talk_failure"

    # Context guard events
    CONTEXT_GUARD_START = "context_guard_start"
    CONTEXT_GUARD_DECISION = "context_guard_decision"
    CONTEXT_GUARD_CACHE = "context_guard_cache"

    # Action queue events
    QUEUE_ENQUEUE = "queue_enqueue"
    QUEUE_DEQUEUE = "queue_dequeue"
    QUEUE_CLEAR = "queue_clear"
    QUEUE_REJECT = "queue_reject"
    
    NOTEBOOK_ENTRY_ADDED = "notebook_entry_added"


class LogLevel(str, Enum):
    """Log levels for event logging"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    SUCCESS = "SUCCESS"


@dataclass
class BotEvent:
    """Structured event data"""
    event_type: EventType
    message: str
    timestamp: float = field(default_factory=time.time)
    level: LogLevel = LogLevel.INFO
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
    
    def __init__(self, debug_mode: bool = True, show_overlay_candidates: bool = False, show_llm_costs: bool = True):
        try:
            self.debug_mode = debug_mode
            self.show_overlay_candidates = show_overlay_candidates
            self.show_llm_costs = show_llm_costs
            self._callbacks: List[Callable[[BotEvent], None]] = []
            self._event_history: List[BotEvent] = []
            self._max_history = 1000
        except Exception:
            # If even initialization fails, set minimal defaults
            self.debug_mode = True
            self.show_overlay_candidates = False
            self.show_llm_costs = True
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
            LogLevel.DEBUG: "🔍",
            LogLevel.INFO: "ℹ️",
            LogLevel.WARNING: "⚠️",
            LogLevel.ERROR: "❌",
            LogLevel.SUCCESS: "✅"
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
    
    def emit(self, event_type: EventType, message: str, level: LogLevel = LogLevel.INFO, **details) -> None:
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
    
    def notebook_entry_added(self, task_id: str, entry_type: NotebookEntryType, entry: Dict[str, Any], **details):
        try:
            self.emit(EventType.NOTEBOOK_ENTRY_ADDED, f"Added notebook entry for task {task_id}", LogLevel.INFO, task_id=task_id, entry_type=entry_type, entry=entry, **details)
        except Exception:
            pass
    
    # Convenience methods - all wrapped in try/except for safety
    def agent_start(self, prompt: str, **details):
        try:
            self.emit(EventType.AGENT_START, f"Started agent with prompt: {prompt}", LogLevel.INFO, prompt=prompt, **details)
        except Exception:
            pass
    
    def agent_complete(self, success: bool, reasoning: str = None, **details):
        try:
            status = "completed successfully" if success else "failed"
            level = LogLevel.SUCCESS if success else LogLevel.ERROR
            msg = f"Agent {status}"
            if reasoning:
                msg += f": {reasoning}"
            self.emit(EventType.AGENT_COMPLETE, msg, level, 
                     success=success, reasoning=reasoning, **details)
        except Exception:
            pass

    def agent_error(self, message: str, **details):
        try:
            self.emit(EventType.AGENT_ERROR, f"Agent error: {message}", LogLevel.ERROR, message=message, **details)
        except Exception:
            pass
    
    def command_generated(self, command: ActionStep, **details):
        try:
            self.emit(EventType.COMMAND_GENERATED, f"Command generated: {command.action}", LogLevel.INFO, command=command.action, **details)
        except Exception:
            pass
    
    def command_start(self, command: str, **details):
        try:
            msg = f"Starting command: {command}"
            self.emit(EventType.COMMAND_START, msg, LogLevel.INFO,
                     command=command, **details)
        except Exception:
            pass

    def command_success(self, command: str, **details):
        try:
            msg = f"Command completed: {command}"
            self.emit(EventType.COMMAND_SUCCESS, msg, LogLevel.SUCCESS, command=command, **details)
        except Exception:
            pass

    def command_failure(self, command: str, error: str = None, **details):
        try:
            msg = f"Command failed: {command}"
            if error:
                msg += f" - {error}"
            self.emit(EventType.COMMAND_FAILURE, msg, LogLevel.ERROR, command=command, error=error, **details)
        except Exception:
            pass
    
    def ask_requested(self, question: str, **details):
        try:
            self.emit(EventType.ASK_REQUESTED, f"Agent asking for help: {question}", LogLevel.INFO, question=question, **details)
        except Exception:
            pass
    
    def ask_command_answered(self, question: str, response: str, **details):
        try:
            self.emit(EventType.ASK_COMMAND_ANSWERED, "User answered question", LogLevel.INFO, question=question, response=response, **details)
        except Exception:
            pass
    
    def ask_command_skipped(self, question: str, **details):
        try:
            self.emit(EventType.ASK_COMMAND_SKIPPED, "User skipped question", LogLevel.INFO, question=question, **details)
        except Exception:
            pass
    
    def ask_command_failure(self, question: str, error: str, **details):
        try:
            self.emit(EventType.ASK_COMMAND_FAILURE, "Error in ask callback", LogLevel.ERROR, question=question, error=error, **details)
        except Exception:
            pass
    
    def agent_talk(self, message: str, **details):
        try:
            self.emit(EventType.AGENT_TALK, "Agent talking", LogLevel.INFO, message=message, **details)
        except Exception:
            pass
    
    def agent_talk_failure(self, message: str, error: str, **details):
        try:
            self.emit(EventType.AGENT_TALK_FAILURE, "Agent talk failed", LogLevel.ERROR, message=message, error=error, **details)
        except Exception:
            pass
    
    def system_info(self, message: str, **details):
        try:
            self.emit(EventType.SYSTEM_INFO, message, LogLevel.INFO, **details)
        except Exception:
            pass
    
    def system_warning(self, message: str, **details):
        try:
            self.emit(EventType.SYSTEM_WARNING, message, LogLevel.WARNING, **details)
        except Exception:
            pass
    
    def system_error(self, message: str, error: Exception = None, **details):
        try:
            msg = message
            if error:
                msg += f" - {str(error)}"
            self.emit(EventType.SYSTEM_ERROR, msg, LogLevel.ERROR, error=str(error) if error else None, **details)
        except Exception:
            pass
    
    def system_debug(self, message: str, **details):
        try:
            self.emit(EventType.SYSTEM_DEBUG, message, LogLevel.DEBUG, **details)
        except Exception:
            pass
    
    def extraction_start(self, prompt: str, **details):
        try:
            self.emit(EventType.EXTRACTION_START, f"Extracting: {prompt}", LogLevel.INFO, prompt=prompt, **details)
        except Exception:
            pass
    
    def extraction_success(self, prompt: str, result: Any = None, **details):
        try:
            self.emit(EventType.EXTRACTION_SUCCESS, f"Extraction completed: {prompt}", LogLevel.SUCCESS, prompt=prompt, result=str(result) if result else None, **details)
        except Exception:
            pass
    
    def extraction_failure(self, prompt: str, error: str = None, **details):
        try:
            msg = f"Extraction failed: {prompt}"
            if error:
                msg += f" - {error}"
            self.emit(EventType.EXTRACTION_FAILURE, msg, LogLevel.ERROR, prompt=prompt, error=error, **details)
        except Exception:
            pass
    
    def plan_cached(self, command: str, **details):
        try:
            self.emit(EventType.PLAN_CACHED, f"Cached plan for command '{command}'", LogLevel.INFO,
                     command=command, **details)
        except Exception:
            pass
    
    def plan_cleared(self, reason: str, **details):
        try:
            self.emit(EventType.PLAN_CLEARED, f"Clearing cached plan ({reason})", LogLevel.INFO, reason=reason, **details)
        except Exception:
            pass

    def action_plan_generation_error(self, error: str, **details):
        try:
            self.emit(EventType.ACTION_PLAN_GENERATION_ERROR, f"Error generating action plan: {error}", LogLevel.ERROR, error=error, **details)
        except Exception:
            pass

    def action_step_generation_error(self, failed_command: str, error: str, **details):
        try:
            self.emit(EventType.ACTION_STEP_GENERATION_ERROR, f"Error generating action step for command '{failed_command}': {error}", LogLevel.ERROR, failed_command=failed_command, error=error, **details)
        except Exception:
            pass

    def overlay_data_snapshot(self, count: int, preview: str = None, **details):
        try:
            msg = f"Overlay data captured: {count} elements"
            if preview:
                msg += f" ({preview})"
            self.emit(EventType.OVERLAY_DATA, msg, LogLevel.DEBUG, count=count, preview=preview, **details)
        except Exception:
            pass

    def plan_generated(self, plan_reasoning: str, confidence: float, expected_outcome: str, steps_summary: str = "", **details):
        try:
            self.emit(
                EventType.PLAN_GENERATED,
                "Plan generated",
                LogLevel.INFO,
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
            self.emit(EventType.PLAN_CACHED, "Reusing cached plan (skipped LLM planning)", LogLevel.INFO, **details)
        except Exception:
            pass
    
    def completion_check(self, is_complete: bool, reasoning: str = None, confidence: float = None, **details):
        try:
            status = "complete" if is_complete else "not complete"
            msg = f"Task {status}"
            if reasoning:
                msg += f": {reasoning}"
            level = LogLevel.SUCCESS if is_complete else LogLevel.INFO
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
            self.emit(EventType.COMPLETION_SUCCESS, msg, LogLevel.SUCCESS, completion_type="agent_signaled", reasoning=reasoning, **details)
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
            self.emit(EventType.ACTION_DETERMINED, msg, LogLevel.INFO, action=action, reasoning=reasoning, confidence=confidence, expected_outcome=expected_outcome, **details)
        except Exception:
            pass
    
    def action_error(self, action: str, error: str = None, **details):
        try:
            msg = f"Action failed: {action}"
            if error:
                msg += f" - {error}"
            self.emit(EventType.ACTION_ERROR, msg, LogLevel.ERROR, action=action, error=error, **details)
        except Exception:
            pass
    
    def extraction_detected(self, prompt: str, **details):
        try:
            self.emit(EventType.EXTRACTION_DETECTED, f"Extraction detected: {prompt}", LogLevel.INFO, prompt=prompt, **details)
        except Exception:
            pass

    def task_decompose_start(self, prompt: str, **details):
        try:
            self.emit(EventType.TASK_DECOMPOSE_START, "Task decomposition started", LogLevel.INFO, prompt=prompt, **details)
        except Exception:
            pass

    def task_decompose_complete(self, tasks: list[TaskDefinition], task_count: int, **details):
        try:
            self.emit(EventType.TASK_DECOMPOSE_COMPLETE, f"Task decomposition complete \n {json.dumps([task.model_dump() for task in tasks], indent=4)}", LogLevel.SUCCESS, task_count=task_count, tasks=tasks, **details)
        except Exception:
            pass

    def task_decompose_fail(self, error: str, **details):
        try:
            self.emit(EventType.TASK_DECOMPOSE_FAIL, f"Task decomposition failed: {error}", LogLevel.ERROR, error=error, **details)
        except Exception:
            pass

    def task_start(self, task_id: str, task: str, **details):
        try:
            self.emit(EventType.TASK_START, f"Task start: {task}", LogLevel.INFO, task_id=task_id, **details)
        except Exception:
            pass

    def task_complete(self, task_id: str, **details):
        try:
            self.emit(EventType.TASK_COMPLETE, f"Task complete: {task_id}", LogLevel.SUCCESS, task_id=task_id, **details)
        except Exception:
            pass

    def task_fail(self, task_id: str, error: str = None, **details):
        try:
            msg = f"Task failed: {task_id}"
            if error:
                msg += f" - {error}"
            self.emit(EventType.TASK_FAIL, msg, LogLevel.ERROR, task_id=task_id, error=error, **details)
        except Exception:
            pass

    def unknown_task_type(self, task_type: str, **details):
        try:
            self.emit(EventType.UNKNOWN_TASK_TYPE, f"Unknown task type: {task_type}", LogLevel.ERROR, task_type=task_type, **details)
        except Exception:
            pass

    def subtask_start(self, instruction: str, **details):
        try:
            self.emit(EventType.SUBTASK_START, f"Subtask start: {instruction}", LogLevel.INFO, instruction=instruction, **details)
        except Exception:
            pass

    def subtask_complete(self, instruction: str, **details):
        try:
            self.emit(EventType.SUBTASK_COMPLETE, f"Subtask complete: {instruction}", LogLevel.SUCCESS, instruction=instruction, **details)
        except Exception:
            pass

    def subtask_fail(self, instruction: str, error: str = None, **details):
        try:
            msg = f"Subtask failed: {instruction}"
            if error:
                msg += f" - {error}"
            self.emit(EventType.SUBTASK_FAIL, msg, LogLevel.ERROR, instruction=instruction, error=error, **details)
        except Exception:
            pass

    def iteration_start(self, task_instruction: str, iteration: int, max_iterations: int, **details):
        try:
            self.emit(EventType.ITERATION_START, f"Iteration start: {iteration}/{max_iterations}", LogLevel.DEBUG, task_instruction=task_instruction, iteration=iteration, max_iterations=max_iterations, **details)
        except Exception:
            pass

    def iteration_complete(self, task_instruction: str, iteration: int, max_iterations: int, **details):
        try:
            self.emit(EventType.ITERATION_COMPLETE, f"Iteration complete: {iteration}/{max_iterations}", LogLevel.DEBUG, task_instruction=task_instruction, iteration=iteration, max_iterations=max_iterations, **details)
        except Exception:
            pass

    def iteration_fail(self, task_instruction: str, iteration: int, error: str = None, **details):
        try:
            msg = f"Iteration failed: {iteration}"
            if error:
                msg += f" - {error}"
            self.emit(EventType.ITERATION_FAIL, msg, LogLevel.WARNING, task_instruction=task_instruction, iteration=iteration, error=error, **details)
        except Exception:
            pass

    def plan_execute_start(self, step_count: int, **details):
        try:
            self.emit(EventType.PLAN_EXECUTE_START, f"Plan execution start ({step_count} steps)", LogLevel.INFO, step_count=step_count, **details)
        except Exception:
            pass

    def plan_execute_complete(self, **details):
        try:
            self.emit(EventType.PLAN_EXECUTE_COMPLETE, "Plan execution completed", LogLevel.SUCCESS, **details)
        except Exception:
            pass

    def plan_execute_fail(self, error: str = None, **details):
        try:
            msg = "Plan execution failed"
            if error:
                msg += f" - {error}"
            self.emit(EventType.PLAN_EXECUTE_FAIL, msg, LogLevel.ERROR, error=error, **details)
        except Exception:
            pass

    def schema_infer_start(self, goal: str, **details):
        try:
            self.emit(EventType.SCHEMA_INFER_START, "Schema inference started", LogLevel.DEBUG, goal=goal, **details)
        except Exception:
            pass

    def schema_infer_success(self, fields: list[str], **details):
        try:
            self.emit(EventType.SCHEMA_INFER_SUCCESS, f"Schema inferred: {fields}", LogLevel.SUCCESS, fields=fields, **details)
        except Exception:
            pass

    def schema_infer_fail(self, error: str, **details):
        try:
            self.emit(EventType.SCHEMA_INFER_FAIL, f"Schema inference failed: {error}", LogLevel.ERROR, error=error, **details)
        except Exception:
            pass

    def schema_validate_fail(self, error: str, **details):
        try:
            self.emit(EventType.SCHEMA_VALIDATE_FAIL, f"Schema validation failed: {error}", LogLevel.WARNING, error=error, **details)
        except Exception:
            pass

    def extraction_retry(self, prompt: str, attempt: int, **details):
        try:
            self.emit(EventType.EXTRACTION_RETRY, f"Extraction retry {attempt}: {prompt}", LogLevel.WARNING, prompt=prompt, attempt=attempt, **details)
        except Exception:
            pass

    def extraction_empty(self, prompt: str, **details):
        try:
            self.emit(EventType.EXTRACTION_EMPTY, f"Extraction empty: {prompt}", LogLevel.WARNING, prompt=prompt, **details)
        except Exception:
            pass

    def agent_pause(self, message: str = None, **details):
        try:
            msg = "Agent paused"
            if message:
                msg += f": {message}"
            self.emit(EventType.AGENT_PAUSE, msg, LogLevel.INFO, message=message, **details)
        except Exception:
            pass

    def agent_resume(self, **details):
        try:
            self.emit(EventType.AGENT_RESUME, "Agent resumed", LogLevel.INFO, **details)
        except Exception:
            pass

    def retry_backoff(self, delay_seconds: float, attempt: int, **details):
        try:
            self.emit(EventType.RETRY_BACKOFF, f"Retry backoff {attempt}: {delay_seconds:.2f}s", LogLevel.WARNING, delay_seconds=delay_seconds, attempt=attempt, **details)
        except Exception:
            pass

    def retry_giveup(self, reason: str = None, **details):
        try:
            msg = "Retry give up"
            if reason:
                msg += f": {reason}"
            self.emit(EventType.RETRY_GIVEUP, msg, LogLevel.ERROR, reason=reason, **details)
        except Exception:
            pass

    def model_fallback(self, primary: str, fallback: str, **details):
        try:
            self.emit(EventType.MODEL_FALLBACK, f"Model fallback: {primary} -> {fallback}", LogLevel.WARNING, primary=primary, fallback=fallback, **details)
        except Exception:
            pass

    def context_guard_start(self, guard_text: str, overlay_index: int = None, **details):
        try:
            self.emit(EventType.CONTEXT_GUARD_START, "Context guard start", LogLevel.DEBUG, guard_text=guard_text, overlay_index=overlay_index, **details)
        except Exception:
            pass

    def context_guard_decision(self, passed: bool, reason: str = None, cached: bool = False, **details):
        try:
            level = LogLevel.SUCCESS if passed else LogLevel.WARNING
            msg = "Context guard passed" if passed else "Context guard failed"
            self.emit(EventType.CONTEXT_GUARD_DECISION, msg, level, passed=passed, reason=reason, cached=cached, **details)
        except Exception:
            pass

    def context_guard_cache(self, guard_text: str, overlay_index: int = None, **details):
        try:
            self.emit(EventType.CONTEXT_GUARD_CACHE, "Context guard cache hit", LogLevel.DEBUG, guard_text=guard_text, overlay_index=overlay_index, **details)
        except Exception:
            pass

    def queue_enqueue(self, action_id: str = None, **details):
        try:
            self.emit(EventType.QUEUE_ENQUEUE, "Queue enqueue", LogLevel.DEBUG, action_id=action_id, **details)
        except Exception:
            pass

    def queue_dequeue(self, action_id: str = None, **details):
        try:
            self.emit(EventType.QUEUE_DEQUEUE, "Queue dequeue", LogLevel.DEBUG, action_id=action_id, **details)
        except Exception:
            pass

    def queue_clear(self, **details):
        try:
            self.emit(EventType.QUEUE_CLEAR, "Queue cleared", LogLevel.INFO, **details)
        except Exception:
            pass

    def queue_reject(self, reason: str, action_id: str = None, **details):
        try:
            msg = f"Queue reject: {reason}"
            self.emit(EventType.QUEUE_REJECT, msg, LogLevel.WARNING, reason=reason, action_id=action_id, **details)
        except Exception:
            pass
    
    def action_coordinates(self, message: str, **details):
        try:
            self.emit(EventType.ACTION_COORDINATES, message, LogLevel.DEBUG, **details)
        except Exception:
            pass
    
    def action_refinement(self, message: str, **details):
        try:
            self.emit(EventType.ACTION_REFINEMENT, message, LogLevel.DEBUG, **details)
        except Exception:
            pass
    
    def llm_cost(self, cost_usd: float, input_tokens: int, output_tokens: int, total_tokens: int, model: str = None, **details):
        try:
            if not self.show_llm_costs:
                return
            msg = f"Prompt Cost: {cost_usd} USD, Input Tokens: {input_tokens}, Output Tokens: {output_tokens}, Total Tokens: {total_tokens}"
            if model:
                msg += f" (Model: {model})"
            self.emit(EventType.LLM_COST, msg, LogLevel.DEBUG, cost_usd=cost_usd, input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=total_tokens, model=model, **details)
        except Exception:
            pass
    
    def command_execution_start(self, instruction: str, target_hint: str = None, **details):
        try:
            msg = f"Executing command for instruction='{instruction}'"
            if target_hint:
                msg += f" target_hint='{target_hint}'"
            self.emit(EventType.COMMAND_EXECUTION_START, msg, LogLevel.DEBUG, instruction=instruction, target_hint=target_hint, **details)
        except Exception:
            pass
    
    def command_execution_complete(self, command: str, success: bool = True, **details):
        try:
            msg = f"Command execution completed: {command}"
            level = LogLevel.SUCCESS if success else LogLevel.ERROR
            self.emit(EventType.COMMAND_EXECUTION_COMPLETE, msg, level, command=command, success=success, **details)
        except Exception:
            pass
    
    def command_execution_failure(self, command: str, error: str = None, **details):
        try:
            msg = f"Command execution failed: {command}"
            if error:
                msg += f" - {error}"
            self.emit(EventType.COMMAND_EXECUTION_FAILURE, msg, LogLevel.ERROR, command=command, error=error, **details)
        except Exception:
            pass
    
    def overlay_selection(self, message: str, **details):
        try:
            self.emit(EventType.OVERLAY_SELECTION, message, LogLevel.DEBUG, **details)
        except Exception:
            pass
    
    def plan_overlay_candidates(self, candidates: list[str], **details):
        try:
            if not self.show_overlay_candidates:
                return
            msg = "Candidate overlays for LLM selection:"
            for candidate in candidates:
                msg += f"\n  • {candidate}"
            self.emit(EventType.PLAN_OVERLAY_CANDIDATES, msg, LogLevel.DEBUG, candidates=candidates, **details)
        except Exception:
            pass
    
    def plan_overlay_chosen(self, overlay_index: int, raw_response: str = None, **details):
        try:
            msg = f"overlay #{overlay_index} chosen"
            if raw_response:
                msg += f" (raw='{raw_response}')"
            self.emit(EventType.PLAN_OVERLAY_CHOSEN, msg, LogLevel.DEBUG, overlay_index=overlay_index, raw_response=raw_response, **details)
        except Exception:
            pass
    
    def command_history(self, command: str, **details):
        try:
            self.emit(EventType.COMMAND_HISTORY, f"Added to command history: '{command}'", LogLevel.DEBUG, command=command, **details)
        except Exception:
            pass
    
    def interaction_recorded(self, interaction_type: str, **details):
        try:
            message = f"Recorded {interaction_type} interaction"
            if details.get('reasoning'):
                message += f"\n   Why: {details['reasoning']}"
            self.emit(EventType.INTERACTION_RECORDED, message, LogLevel.DEBUG, interaction_type=interaction_type, **details)
        except Exception:
            pass
    
    def action_state_change(self, message: str, url_before: str = None, url_after: str = None, dom_changed: bool = None, **details):
        try:
            self.emit(EventType.ACTION_STATE_CHANGE, message, LogLevel.INFO, url_before=url_before, url_after=url_after, dom_changed=dom_changed, **details)
        except Exception:
            pass
    
    def action_state_change_failure(self, message: str, **details):
        try:
            self.emit(EventType.ACTION_STATE_CHANGE_FAILURE, message, LogLevel.WARNING, **details)
        except Exception:
            pass

# Global instance
_global_event_logger: Optional[EventLogger] = None

def get_event_logger() -> EventLogger:
    """Get the global event logger instance"""
    global _global_event_logger
    if _global_event_logger is None:
        _global_event_logger = EventLogger(debug_mode=True, show_overlay_candidates=False, show_llm_costs=True)  # Default to debug for backward compatibility
    return _global_event_logger

def set_event_logger(logger: EventLogger) -> None:
    """Set the global event logger instance"""
    global _global_event_logger
    _global_event_logger = logger
