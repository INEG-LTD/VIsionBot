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
from utils.debug_print import dprint


class EventType(str, Enum):
    """All event types that can be logged."""

    # Agent lifecycle
    AGENT_START = "agent_start"
    AGENT_COMPLETE = "agent_complete"
    AGENT_ERROR = "agent_error"

    # Mission execution
    ITERATION_START = "iteration_start"
    ITERATION_COMPLETE = "iteration_complete"
    ACTION_DETERMINED = "action_determined"
    ACTION_COMPLETE = "action_complete"
    ACTION_ERROR = "action_error"
    STUCK_DETECTED = "stuck_detected"
    LOOP_STATE_CHANGED = "loop_state_changed"
    CHECKPOINT_CHANGED = "checkpoint_changed"
    ELEMENT_CAPTURE = "element_capture"
    SCREENSHOT_CAPTURED = "screenshot_captured"
    BROWSER_NAVIGATION = "browser_navigation"

    # Command execution
    COMMAND_START = "command_start"
    COMMAND_SUCCESS = "command_success"
    COMMAND_FAILURE = "command_failure"
    COMMAND_EXECUTION_START = "command_execution_start"
    COMMAND_EXECUTION_COMPLETE = "command_execution_complete"
    COMMAND_EXECUTION_FAILURE = "command_execution_failure"
    COMMAND_HISTORY = "command_history"

    # User interaction
    ASK_REQUESTED = "ask_requested"
    ASK_COMMAND_ANSWERED = "ask_command_answered"
    ASK_COMMAND_SKIPPED = "ask_command_skipped"
    ASK_COMMAND_FAILURE = "ask_command_failure"

    # System logging
    SYSTEM_INFO = "system_info"
    SYSTEM_WARNING = "system_warning"
    SYSTEM_ERROR = "system_error"
    SYSTEM_DEBUG = "system_debug"
    SANDBOX_DECISION = "sandbox_decision"
    SANDBOX_BLOCKED_ACTION = "sandbox_blocked_action"
    CLEANUP_START = "cleanup_start"
    CLEANUP_DELETION = "cleanup_deletion"
    CLEANUP_COMPLETE = "cleanup_complete"

    # Extraction
    EXTRACTION_START = "extraction_start"
    EXTRACTION_SUCCESS = "extraction_success"
    EXTRACTION_FAILURE = "extraction_failure"
    EXTRACTION_DETECTED = "extraction_detected"
    EXTRACTION_RETRY = "extraction_retry"
    EXTRACTION_EMPTY = "extraction_empty"

    # Element selection
    OVERLAY_SELECTION = "overlay_selection"
    OVERLAY_DATA = "overlay_data"
    PLAN_OVERLAY_CANDIDATES = "plan_overlay_candidates"
    PLAN_OVERLAY_CHOSEN = "plan_overlay_chosen"
    ACTION_REFINEMENT = "action_refinement"

    # Queue
    QUEUE_ENQUEUE = "queue_enqueue"
    QUEUE_DEQUEUE = "queue_dequeue"
    QUEUE_CLEAR = "queue_clear"
    QUEUE_REJECT = "queue_reject"

    # Cost tracking
    LLM_COST = "llm_cost"


class LogLevel(str, Enum):
    """Log levels for event logging."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    SUCCESS = "SUCCESS"


@dataclass
class BotEvent:
    """Structured event data."""
    event_type: EventType
    message: str
    timestamp: float = field(default_factory=time.time)
    level: LogLevel = LogLevel.INFO
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type.value,
            "message": self.message,
            "timestamp": self.timestamp,
            "timestamp_iso": datetime.fromtimestamp(self.timestamp).isoformat(),
            "level": self.level,
            "details": self.details,
        }


class EventLogger:
    """
    Simple, robust event logger.

    In debug mode: prints directly to console.
    In normal mode: only calls callbacks (no prints).
    """

    def __init__(self, debug_mode: bool = True, show_overlay_candidates: bool = False, show_llm_costs: bool = True):
        try:
            self.debug_mode = debug_mode
            self.show_overlay_candidates = show_overlay_candidates
            self.show_llm_costs = show_llm_costs
            self._callbacks: List[Callable[[BotEvent], None]] = []
            self._typed_callbacks: Dict[EventType, List[Callable[[BotEvent], None]]] = {}
            self._event_history: List[BotEvent] = []
            self._max_history = 1000
            self._total_cost_usd: float = 0.0
            self._total_tokens: int = 0
        except Exception:
            self.debug_mode = True
            self.show_overlay_candidates = False
            self.show_llm_costs = True
            self._callbacks = []
            self._typed_callbacks = {}
            self._event_history = []
            self._max_history = 1000
            self._total_cost_usd = 0.0
            self._total_tokens = 0

    def register_callback(self, callback: Callable[[BotEvent], None]) -> None:
        if callback not in self._callbacks:
            self._callbacks.append(callback)

    def unregister_callback(self, callback: Callable[[BotEvent], None]) -> None:
        if callback in self._callbacks:
            self._callbacks.remove(callback)

    def on(self, event_type: EventType, callback: Callable[[BotEvent], None]) -> None:
        callbacks = self._typed_callbacks.setdefault(event_type, [])
        if callback not in callbacks:
            callbacks.append(callback)

    def off(self, event_type: EventType, callback: Callable[[BotEvent], None]) -> None:
        callbacks = self._typed_callbacks.get(event_type, [])
        if callback in callbacks:
            callbacks.remove(callback)
        if not callbacks and event_type in self._typed_callbacks:
            del self._typed_callbacks[event_type]

    @property
    def total_cost_usd(self) -> float:
        return self._total_cost_usd

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    def set_usage_totals(self, *, total_cost_usd: float, total_tokens: int) -> None:
        """Restore aggregate usage counters from checkpoint state."""
        try:
            self._total_cost_usd = max(0.0, float(total_cost_usd or 0.0))
            self._total_tokens = max(0, int(total_tokens or 0))
        except Exception:
            self._total_cost_usd = 0.0
            self._total_tokens = 0

    def get_event_history(self) -> List[BotEvent]:
        return list(self._event_history)

    def _safe_emit(self, event: BotEvent) -> None:
        try:
            self._event_history.append(event)
            if len(self._event_history) > self._max_history:
                self._event_history.pop(0)
        except Exception:
            pass

        if self.debug_mode:
            try:
                self._print_event(event)
            except Exception:
                pass

        for callback in list(self._callbacks):
            try:
                callback(event)
            except Exception:
                pass

        for callback in list(self._typed_callbacks.get(event.event_type, [])):
            try:
                callback(event)
            except Exception:
                pass

    def _print_event(self, event: BotEvent) -> None:
        level_emoji = {
            LogLevel.DEBUG: "🔍",
            LogLevel.INFO: "ℹ️",
            LogLevel.WARNING: "⚠️",
            LogLevel.ERROR: "❌",
            LogLevel.SUCCESS: "✅",
        }
        emoji = level_emoji.get(event.level, "•")
        dprint(f"{emoji} {event.message}")

        if event.details:
            for key, value in event.details.items():
                if value is not None and key not in ("timestamp", "timestamp_iso"):
                    try:
                        if isinstance(value, (str, int, float, bool)):
                            dprint(f"   {key}: {value}")
                    except Exception:
                        pass

    def emit(self, event_type: EventType, message: str, level: LogLevel = LogLevel.INFO, **details) -> None:
        try:
            event = BotEvent(event_type=event_type, message=message, level=level, details=details)
            self._safe_emit(event)
        except Exception:
            if self.debug_mode:
                try:
                    dprint(f"⚠️ Event logger error: {message}")
                except Exception:
                    pass

    # ── Agent lifecycle ──────────────────────────────────────────────────

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
            self.emit(EventType.AGENT_COMPLETE, msg, level, success=success, reasoning=reasoning, **details)
        except Exception:
            pass

    def agent_error(self, message: str, **details):
        try:
            self.emit(EventType.AGENT_ERROR, f"Agent error: {message}", LogLevel.ERROR, message=message, **details)
        except Exception:
            pass

    # ── Mission execution ────────────────────────────────────────────────

    def iteration_start(self, iteration: int, max_iterations: int, **details):
        try:
            self.emit(EventType.ITERATION_START, f"Iteration start: {iteration}/{max_iterations}", LogLevel.DEBUG, iteration=iteration, max_iterations=max_iterations, **details)
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

    def action_complete(
        self,
        tool: str,
        narrative: str,
        success: bool,
        result_str: str,
        duration_ms: float,
        **details,
    ) -> None:
        try:
            level = LogLevel.SUCCESS if success else LogLevel.ERROR
            message = f"Action complete: {tool} -> {result_str}"
            if narrative:
                message += f" | {narrative}"
            self.emit(
                EventType.ACTION_COMPLETE,
                message,
                level,
                tool=tool,
                narrative=narrative,
                success=success,
                result=result_str,
                duration_ms=duration_ms,
                **details,
            )
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

    def loop_state_changed(
        self,
        change: str,
        loop_round: int,
        loop_count: Optional[int],
        loop_description: str,
        **details,
    ) -> None:
        try:
            msg = f"Loop {change}: round={loop_round}, total={loop_count or '?'}"
            if loop_description:
                msg += f" ({loop_description})"
            self.emit(
                EventType.LOOP_STATE_CHANGED,
                msg,
                LogLevel.INFO,
                change=change,
                loop_round=loop_round,
                loop_count=loop_count,
                loop_description=loop_description,
                **details,
            )
        except Exception:
            pass

    def iteration_complete(self, iteration: int, duration_ms: float, **details) -> None:
        try:
            self.emit(
                EventType.ITERATION_COMPLETE,
                f"Iteration complete: {iteration} ({duration_ms:.1f} ms)",
                LogLevel.DEBUG,
                iteration=iteration,
                duration_ms=duration_ms,
                **details,
            )
        except Exception:
            pass

    def checkpoint_changed(self, pending: bool, **details) -> None:
        try:
            self.emit(
                EventType.CHECKPOINT_CHANGED,
                f"Checkpoint {'pending' if pending else 'cleared'}",
                LogLevel.DEBUG,
                pending=pending,
                **details,
            )
        except Exception:
            pass

    def element_capture(self, total: int, text_rich: int, text_poor: int, **details) -> None:
        try:
            self.emit(
                EventType.ELEMENT_CAPTURE,
                f"Captured elements: total={total}, text_rich={text_rich}, text_poor={text_poor}",
                LogLevel.DEBUG,
                total=total,
                text_rich=text_rich,
                text_poor=text_poor,
                **details,
            )
        except Exception:
            pass

    def screenshot_captured(
        self,
        screenshot_id: str,
        byte_size: int,
        sha256: str,
        *,
        iteration: Optional[int] = None,
        path: Optional[str] = None,
        in_memory: bool = True,
        url: Optional[str] = None,
        **details,
    ) -> None:
        try:
            msg = f"Screenshot captured: {screenshot_id} ({byte_size} bytes)"
            self.emit(
                EventType.SCREENSHOT_CAPTURED,
                msg,
                LogLevel.DEBUG,
                screenshot_id=screenshot_id,
                iteration=iteration,
                byte_size=byte_size,
                sha256=sha256,
                path=path,
                in_memory=in_memory,
                url=url,
                **details,
            )
        except Exception:
            pass

    def browser_navigation(self, url: str, **details) -> None:
        try:
            self.emit(
                EventType.BROWSER_NAVIGATION,
                f"Browser navigated: {url}",
                LogLevel.INFO,
                url=url,
                **details,
            )
        except Exception:
            pass

    # ── Command execution ────────────────────────────────────────────────

    def command_start(self, command: str, **details):
        try:
            self.emit(EventType.COMMAND_START, f"Starting command: {command}", LogLevel.INFO, command=command, **details)
        except Exception:
            pass

    def command_success(self, command: str, **details):
        try:
            self.emit(EventType.COMMAND_SUCCESS, f"Command completed: {command}", LogLevel.SUCCESS, command=command, **details)
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
            level = LogLevel.SUCCESS if success else LogLevel.ERROR
            self.emit(EventType.COMMAND_EXECUTION_COMPLETE, f"Command execution completed: {command}", level, command=command, success=success, **details)
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

    def command_history(self, command: str, **details):
        try:
            self.emit(EventType.COMMAND_HISTORY, f"Added to command history: '{command}'", LogLevel.DEBUG, command=command, **details)
        except Exception:
            pass

    # ── User interaction (ask) ───────────────────────────────────────────

    def ask_requested(
        self,
        question: str,
        options: Optional[List[str]] = None,
        multi_select: bool = False,
        yes_no: bool = False,
        **details,
    ):
        try:
            self.emit(
                EventType.ASK_REQUESTED,
                f"Agent asking for help: {question}",
                LogLevel.INFO,
                question=question,
                options=list(options or []),
                multi_select=bool(multi_select),
                yes_no=bool(yes_no),
                **details,
            )
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

    # ── System logging ───────────────────────────────────────────────────

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

    def sandbox_decision(
        self,
        *,
        check_type: str,
        target: str,
        allowed: bool,
        reason: str = "",
        **details,
    ) -> None:
        try:
            level = LogLevel.DEBUG if allowed else LogLevel.WARNING
            msg = f"Sandbox decision: {check_type} -> {'allowed' if allowed else 'blocked'}"
            if reason:
                msg += f" ({reason})"
            self.emit(
                EventType.SANDBOX_DECISION,
                msg,
                level,
                check_type=check_type,
                target=target,
                allowed=allowed,
                reason=reason,
                **details,
            )
        except Exception:
            pass

    def sandbox_blocked_action(
        self,
        *,
        check_type: str,
        target: str,
        reason: str,
        **details,
    ) -> None:
        try:
            self.emit(
                EventType.SANDBOX_BLOCKED_ACTION,
                f"Sandbox blocked action: {check_type} ({reason})",
                LogLevel.WARNING,
                check_type=check_type,
                target=target,
                reason=reason,
                **details,
            )
        except Exception:
            pass

    def cleanup_start(self, *, root: str, **details) -> None:
        try:
            self.emit(
                EventType.CLEANUP_START,
                f"Storage cleanup start: {root}",
                LogLevel.INFO,
                root=root,
                **details,
            )
        except Exception:
            pass

    def cleanup_deletion(self, *, path: str, reason: str, bytes_removed: int = 0, **details) -> None:
        try:
            self.emit(
                EventType.CLEANUP_DELETION,
                f"Storage cleanup deletion: {path} ({reason})",
                LogLevel.INFO,
                path=path,
                reason=reason,
                bytes_removed=bytes_removed,
                **details,
            )
        except Exception:
            pass

    def cleanup_complete(self, *, root: str, **details) -> None:
        try:
            self.emit(
                EventType.CLEANUP_COMPLETE,
                f"Storage cleanup complete: {root}",
                LogLevel.INFO,
                root=root,
                **details,
            )
        except Exception:
            pass

    # ── Extraction ───────────────────────────────────────────────────────

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

    def extraction_detected(self, prompt: str, **details):
        try:
            self.emit(EventType.EXTRACTION_DETECTED, f"Extraction detected: {prompt}", LogLevel.INFO, prompt=prompt, **details)
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

    # ── Element selection ────────────────────────────────────────────────

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

    def action_refinement(self, message: str, **details):
        try:
            self.emit(EventType.ACTION_REFINEMENT, message, LogLevel.DEBUG, **details)
        except Exception:
            pass

    # ── Queue ────────────────────────────────────────────────────────────

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
            self.emit(EventType.QUEUE_REJECT, f"Queue reject: {reason}", LogLevel.WARNING, reason=reason, action_id=action_id, **details)
        except Exception:
            pass

    # ── Cost tracking ────────────────────────────────────────────────────

    def llm_cost(self, cost_usd: float, input_tokens: int, output_tokens: int, total_tokens: int, model: str = None, **details):
        try:
            self._total_cost_usd += float(cost_usd or 0.0)
            self._total_tokens += int(total_tokens or 0)
            if not self.show_llm_costs:
                return
            msg = f"Prompt Cost: {cost_usd} USD, Input Tokens: {input_tokens}, Output Tokens: {output_tokens}, Total Tokens: {total_tokens}"
            if model:
                msg += f" (Model: {model})"
            self.emit(EventType.LLM_COST, msg, LogLevel.DEBUG, cost_usd=cost_usd, input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=total_tokens, model=model, **details)
        except Exception:
            pass


# Global instance
_global_event_logger: Optional[EventLogger] = None


def get_event_logger() -> EventLogger:
    """Get the global event logger instance."""
    global _global_event_logger
    if _global_event_logger is None:
        _global_event_logger = EventLogger(debug_mode=True, show_overlay_candidates=False, show_llm_costs=True)
    return _global_event_logger


def set_event_logger(logger: EventLogger) -> None:
    """Set the global event logger instance."""
    global _global_event_logger
    _global_event_logger = logger
