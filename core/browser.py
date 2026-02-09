"""
Vision Bot - Clean modular version.
"""
from __future__ import annotations

import re
import time
from typing import Any, Optional, List, Dict

# Compatible import for playwright_stealth across versions
try:
    from playwright_stealth import Stealth
    stealth_instance = Stealth()

    def stealth_sync(page):
        return stealth_instance.apply_stealth_sync(page)

except ImportError:
    # Fallback: disable stealth if not available
    def stealth_sync(page):
        pass  # No-op function

from models.models import PageInfo
# Removed goal imports - goals system no longer used
from utils.debug_print import dprint
from core.config import Config
from browser.provider import BrowserProvider, create_browser_provider
from lib.errors import (
    BotNotStartedError,
    BotTerminatedError,
    ActionFailedError,
    ValidationError,
    ErrorContext,
)

DOM_ELEMENT_CENTER_SCRIPT = """
(idx) => {
    const el = document.querySelector(`[data-dom-index="${idx}"]`);
    if (!el) {
        return null;
    }
    const rect = el.getBoundingClientRect();
    if (!rect || rect.width === 0 || rect.height === 0) {
        return null;
    }
    return {
        x: Math.round(rect.left + rect.width / 2),
        y: Math.round(rect.top + rect.height / 2),
    };
}
"""

DOM_ELEMENT_SCROLL_INTO_VIEW_SCRIPT = """
(idx) => {
    const el = document.querySelector(`[data-dom-index="${idx}"]`);
    if (!el) {
        return null;
    }
    if (el.scrollIntoView) {
        el.scrollIntoView({block: "center", inline: "center"});
    }
    const rect = el.getBoundingClientRect();
    if (!rect) {
        return null;
    }
    return {
        x: rect.x,
        y: rect.y,
        width: rect.width,
        height: rect.height,
    };
}
"""


class ExecutionTimer:
    """Tracks execution timings for tasks, iterations, and actions"""
    
    def __init__(self):
        self.task_start_time: Optional[float] = None
        self.task_end_time: Optional[float] = None
        self.iterations: List[Dict[str, float]] = []  # List of {start, end} dicts
        self.actions: List[Dict[str, Any]] = []  # List of {action_id, command, start, end} dicts
        self.current_iteration_start: Optional[float] = None
        self.current_action_id: Optional[str] = None
        self.current_action_start: Optional[float] = None
        self._current_command_text: str = ""
    
    def start_task(self) -> None:
        """Start tracking task execution"""
        self.task_start_time = time.time()
        self.iterations = []
        self.actions = []
    
    def end_task(self) -> None:
        """End task tracking"""
        self.task_end_time = time.time()
        # End any active iteration or action
        if self.current_iteration_start is not None:
            self.end_iteration()
        if self.current_action_start is not None:
            self.end_action()
    
    def start_iteration(self) -> None:
        """Start tracking an iteration"""
        # End previous iteration if still active
        if self.current_iteration_start is not None:
            self.end_iteration()
        self.current_iteration_start = time.time()
    
    def end_iteration(self) -> None:
        """End current iteration tracking"""
        if self.current_iteration_start is not None:
            self.iterations.append({
                "start": self.current_iteration_start,
                "end": time.time()
            })
            self.current_iteration_start = None
    
    def start_action(self, action_id: str, command: str) -> None:
        """Start tracking an action"""
        # End previous action if still active
        if self.current_action_start is not None:
            self.end_action()
        self.current_action_id = action_id
        self.current_action_start = time.time()
        self._current_command_text = command
    
    def end_action(self) -> None:
        """End current action tracking"""
        if self.current_action_start is not None and self.current_action_id is not None:
            self.actions.append({
                "action_id": self.current_action_id,
                "command": getattr(self, "_current_command_text", ""),
                "start": self.current_action_start,
                "end": time.time()
            })
            self.current_action_id = None
            self.current_action_start = None
            self._current_command_text = ""
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all timings"""
        summary = {
            "task": {},
            "iterations": [],
            "actions": []
        }
        
        # Task timing
        if self.task_start_time and self.task_end_time:
            task_duration = self.task_end_time - self.task_start_time
            summary["task"] = {
                "duration_seconds": round(task_duration, 2),
                "duration_formatted": self._format_duration(task_duration)
            }
        
        # Iteration timings
        for i, iter_data in enumerate(self.iterations, 1):
            duration = iter_data["end"] - iter_data["start"]
            summary["iterations"].append({
                "iteration": i,
                "duration_seconds": round(duration, 2),
                "duration_formatted": self._format_duration(duration)
            })
        
        # Action timings
        for action_data in self.actions:
            duration = action_data["end"] - action_data["start"]
            summary["actions"].append({
                "action_id": action_data["action_id"],
                "command": action_data.get("command", ""),
                "duration_seconds": round(duration, 2),
                "duration_formatted": self._format_duration(duration)
            })
        
        return summary
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in a human-readable way"""
        if seconds < 1:
            return f"{int(seconds * 1000)}ms"
        elif seconds < 60:
            return f"{seconds:.2f}s"
        else:
            mins = int(seconds // 60)
            secs = seconds % 60
            return f"{mins}m {secs:.2f}s"
    
    def log_summary(self, event_logger=None) -> None:
        """Log timing summary to console"""
        summary = self.get_summary()
        
        try:
            # Use event_logger if provided, otherwise use print
            log_func = event_logger.system_info if event_logger else print
            
            log_func("\n" + "="*60)
            log_func("⏱️  EXECUTION TIMING SUMMARY")
            log_func("="*60)
            
            # Task timing
            if summary["task"]:
                log_func(f"\n📋 Task Duration: {summary['task']['duration_formatted']} ({summary['task']['duration_seconds']}s)")
            
            # Iteration timings
            if summary["iterations"]:
                total_iter_time = sum(iter_data["duration_seconds"] for iter_data in summary["iterations"])
                avg_iter_time = total_iter_time / len(summary["iterations"])
                log_func(f"\n🔄 Iterations: {len(summary['iterations'])}")
                log_func(f"   Total iteration time: {self._format_duration(total_iter_time)}")
                log_func(f"   Average per iteration: {self._format_duration(avg_iter_time)}")
                fastest_iter = min(summary['iterations'], key=lambda x: x['duration_seconds'])
                slowest_iter = max(summary['iterations'], key=lambda x: x['duration_seconds'])
                log_func(f"   Fastest iteration: {fastest_iter['duration_formatted']}")
                log_func(f"   Slowest iteration: {slowest_iter['duration_formatted']}")
            
            # Action timings
            if summary["actions"]:
                total_action_time = sum(action_data["duration_seconds"] for action_data in summary["actions"])
                avg_action_time = total_action_time / len(summary["actions"])
                log_func(f"\n🎯 Actions: {len(summary['actions'])}")
                log_func(f"   Total action time: {self._format_duration(total_action_time)}")
                log_func(f"   Average per action: {self._format_duration(avg_action_time)}")
                
                # Show top 5 slowest actions
                sorted_actions = sorted(summary["actions"], key=lambda x: x["duration_seconds"], reverse=True)
                log_func("\n   Top 5 slowest actions:")
                for i, action in enumerate(sorted_actions[:5], 1):
                    command_text = action["command"][:50] + "..." if len(action.get("command", "")) > 50 else action.get("command", "")
                    log_func(f"   {i}. {action['action_id']}: {action['duration_formatted']} - {command_text}")
            
            log_func("="*60 + "\n")
        except Exception:
            pass


class Browser:
    """Modular vision-based web automation bot"""


    def __init__(
        self,
        config: Optional[Config] = None,
        browser_provider: Optional[BrowserProvider] = None,
    ):
        # Create default config if not provided
        if config is None:
            config = Config()
        
        # Store config for later use
        self.config = config
        self.browser_provider = browser_provider
        # Handle browser provider
        if self.browser_provider is None:
            # Create provider from config
            self.browser_provider = create_browser_provider(self.config.browser)
        
        self.page = self.browser_provider.get_page() 
        
        # Set the centralized model configuration
        self.started = False
        
        # Interceptor registry (will be passed to Agent)
        self.interceptors: List[Dict[str, Any]] = []

        # Bot termination state
        self.terminated = False

    def end(self) -> None:
        """
        Terminate the bot and prevent any subsequent operations.
        """
        if self.terminated:
            try:
                self.event_logger.system_warning("Bot is already terminated")
            except Exception:
                pass
            return
            
        try:
            self.event_logger.system_info("Terminating bot...")
        except Exception:
            pass
        
        # Close browser provider and cleanup
        try:
            if hasattr(self, 'browser_provider') and self.browser_provider:
                try:
                    self.event_logger.system_info("Closing browser...")
                except Exception:
                    pass
                self.browser_provider.close()
        except Exception as e:
            try:
                self.event_logger.system_error("Error closing browser", error=e)
            except Exception:
                pass
        
        # Mark as terminated
        self.terminated = True
        self.started = False
        
        try:
            self.event_logger.system_info("Bot terminated successfully")
        except Exception:
            pass

    def _check_termination(self) -> None:
        """
        Check if bot is terminated and raise error if so.
        
        Raises:
            BotTerminatedError: If bot has been terminated
        """
        if self.terminated:
            raise BotTerminatedError(
                "Bot has been terminated. No further operations are allowed.",
                context=ErrorContext(
                    error_type="BotTerminatedError",
                    message="Bot has been terminated. No further operations are allowed.",
                    metadata={"terminated": True}
                )
            )
    
    def goto(self, url: str, timeout: int = 2000) -> None:
        """Go to a URL"""
        self._check_termination()
        
        if not self.started:
            dprint("❌ Bot not started")
            return
        
        self.page.goto(url, wait_until="domcontentloaded", timeout=timeout)
        self.url = url
        # Ensure SessionTracker history reflects the first real navigation instead of about:blank
        try:
            if hasattr(self, 'session_tracker') and self.session_tracker:
                hist = getattr(self.session_tracker, 'url_history', None)
                ptr = getattr(self.session_tracker, 'url_pointer', None)
                current = self.page.url
                # If we only have the initial about:blank entry, replace it with the real URL
                if isinstance(hist, list) and len(hist) == 1 and (hist[0] or '').startswith('about:blank'):
                    self.session_tracker.url_history = [current]
                    self.session_tracker.url_pointer = 0
                # If history exists but pointer is not at the end, truncate forward stack and append
                elif isinstance(hist, list) and isinstance(ptr, int) and 0 <= ptr < len(hist):
                    if hist[ptr] != current:
                        # Truncate any forward entries
                        if ptr < (len(hist) - 1):
                            self.session_tracker.url_history = hist[: ptr + 1]
                        # Append only if it's not already the last entry
                        if not self.session_tracker.url_history or self.session_tracker.url_history[-1] != current:
                            self.session_tracker.url_history.append(current)
                        self.session_tracker.url_pointer = len(self.session_tracker.url_history) - 1
        except Exception:
            # Non-fatal: history sync is best-effort
            pass

    def _scroll_overlay_into_view(self, overlay_index: int) -> Optional[Dict[str, Any]]:
        if overlay_index is None or self.page is None:
            return None
        try:
            return self.page.evaluate(DOM_ELEMENT_SCROLL_INTO_VIEW_SCRIPT, overlay_index)
        except Exception:
            return None
    
    # ==================== Convenience Methods ====================

    def wait_for_load(self, timeout: int = 30000, state: str = "networkidle") -> None:
        """
        Wait for the page to finish loading.
        
        Args:
            timeout: Maximum time to wait in milliseconds (default: 30000)
            state: Load state to wait for: "load", "domcontentloaded", "networkidle" (default: "networkidle")
            
        Raises:
            BotTerminatedError: If bot has been terminated
            BotNotStartedError: If bot is not started
            ValidationError: If invalid state is provided
        """
        self._check_termination()
        if not self.started or not self.page:
            raise BotNotStartedError(
                "Bot not started. Call bot.start() first.",
                context=ErrorContext(
                    error_type="BotNotStartedError",
                    message="Bot not started. Call bot.start() first."
                )
            )
        
        valid_states = ["load", "domcontentloaded", "networkidle"]
        if state not in valid_states:
            raise ValidationError(
                f"Invalid state: {state}. Must be one of {valid_states}",
                context=ErrorContext(
                    error_type="ValidationError",
                    message=f"Invalid load state: {state}",
                    action_data={"state": state, "valid_states": valid_states, "timeout": timeout}
                )
            )
        
        try:
            self.page.wait_for_load_state(state, timeout=timeout)
        except Exception as e:
            # Don't raise, just log - sometimes pages don't fully load
            try:
                self.event_logger.system_warning(f"Page load wait timeout or error: {e}")
            except Exception:
                pass
