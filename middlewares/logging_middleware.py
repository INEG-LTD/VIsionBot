"""Logging middleware for BrowserVisionBot."""

from middleware import Middleware, ActionContext
from typing import Any
from utils.debug_print import dprint, PrintMode


class LoggingMiddleware(Middleware):
    """
    Logs all actions to console.
    
    Example:
        >>> bot.use(LoggingMiddleware())
        🔵 Starting: click
        ✅ Completed: click
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize logging middleware.
        
        Args:
            verbose: If True, log detailed information
        """
        self.verbose = verbose
    
    def before_action(self, context: ActionContext) -> ActionContext:
        """Log action start."""
        if self.verbose:
            dprint(f"🔵 Starting: {context.action_type}")
            if context.action_data:
                dprint(f"   Data: {context.action_data}")
        else:
            dprint(f"🔵 {context.action_type}")
        return context
    
    def after_action(self, context: ActionContext, result: Any) -> Any:
        """Log action completion."""
        if self.verbose:
            dprint(f"✅ Completed: {context.action_type}")
        else:
            dprint(f"✅ {context.action_type}")
        return result
    
    def on_error(self, context: ActionContext, error: Exception) -> None:
        """Log errors."""
        dprint(f"❌ Error in {context.action_type}: {error}")
