"""
Simple, robust event-driven logging system for BrowserVisionBot.

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
    
    # Goal events
    GOAL_START = "goal_start"
    GOAL_SUCCESS = "goal_success"
    GOAL_FAILURE = "goal_failure"
    
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
    
    # Interaction tracking
    INTERACTION_RECORDED = "interaction_recorded"
    
    # Action state details
    ACTION_STATE_CHANGE = "action_state_change"


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
    
    def __init__(self, debug_mode: bool = True):
        try:
            self.debug_mode = debug_mode
            self._callbacks: List[Callable[[BotEvent], None]] = []
            self._event_history: List[BotEvent] = []
            self._max_history = 1000
        except Exception:
            # If even initialization fails, set minimal defaults
            self.debug_mode = True
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
        print(f"{emoji} {event.message}")
        
        # Print important details
        if event.details:
            for key, value in event.details.items():
                if value is not None and key not in ['timestamp', 'timestamp_iso']:
                    # Only print simple types to avoid errors
                    try:
                        if isinstance(value, (str, int, float, bool)):
                            print(f"   {key}: {value}")
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
                    print(f"⚠️ Event logger error: {message}")
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
    
    def goal_start(self, goal_description: str, command_id: str = None, **details):
        try:
            msg = f"Starting goal: {goal_description}"
            if command_id:
                msg += f" [ID: {command_id}]"
            self.emit(EventType.GOAL_START, msg, "INFO", 
                     goal_description=goal_description, command_id=command_id, **details)
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
    
    def plan_cached(self, goal_description: str, **details):
        try:
            self.emit(EventType.PLAN_CACHED, f"Cached plan for goal '{goal_description}'", "INFO", 
                     goal_description=goal_description, **details)
        except Exception:
            pass
    
    def plan_cleared(self, reason: str, **details):
        try:
            self.emit(EventType.PLAN_CLEARED, f"Clearing cached plan ({reason})", "INFO", reason=reason, **details)
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
    
    def command_execution_complete(self, goal_description: str, success: bool = True, **details):
        try:
            msg = f"Command execution completed: {goal_description}"
            level = "SUCCESS" if success else "ERROR"
            self.emit(EventType.COMMAND_EXECUTION_COMPLETE, msg, level, goal_description=goal_description, success=success, **details)
        except Exception:
            pass
    
    def overlay_selection(self, message: str, **details):
        try:
            self.emit(EventType.OVERLAY_SELECTION, message, "DEBUG", **details)
        except Exception:
            pass
    
    def plan_overlay_candidates(self, candidates: List[str], **details):
        try:
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
        _global_event_logger = EventLogger(debug_mode=True)  # Default to debug for backward compatibility
    return _global_event_logger

def set_event_logger(logger: EventLogger) -> None:
    """Set the global event logger instance"""
    global _global_event_logger
    _global_event_logger = logger
