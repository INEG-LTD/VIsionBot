"""Retry middleware for Browser."""

import time
from middleware.base import Middleware, ActionContext
from typing import Any
from utils.debug_print import dprint, PrintMode
from utils.event_logger import get_event_logger


class RetryMiddleware(Middleware):
    """
    Automatically retry failed actions.
    
    Example:
        >>> bot.use(RetryMiddleware(max_retries=3, backoff=2.0))
        # Will retry up to 3 times with exponential backoff
    """
    
    def __init__(self, max_retries: int = 3, backoff: float = 2.0):
        """
        Initialize retry middleware.
        
        Args:
            max_retries: Maximum number of retry attempts
            backoff: Backoff multiplier (exponential backoff)
        """
        self.max_retries = max_retries
        self.backoff = backoff
    
    def on_error(self, context: ActionContext, error: Exception) -> None:
        """Handle error and potentially retry."""
        retries = context.metadata.get('retries', 0)
        
        if retries < self.max_retries:
            # Calculate backoff time
            wait_time = self.backoff ** retries
            
            dprint(f"⚠️  Retry {retries + 1}/{self.max_retries} after {wait_time:.1f}s...")
            get_event_logger().retry_backoff(wait_time, retries + 1)
            time.sleep(wait_time)
            
            # Mark for retry
            context.metadata['retries'] = retries + 1
            context.metadata['should_retry'] = True
        else:
            dprint(f"❌ Max retries ({self.max_retries}) exceeded")
            get_event_logger().retry_giveup("max_retries_exceeded", max_retries=self.max_retries)
            context.metadata['should_retry'] = False
