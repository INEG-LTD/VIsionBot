"""Retry middleware for BrowserVisionBot."""

import time
from middleware import Middleware, ActionContext
from typing import Any


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
            
            print(f"⚠️  Retry {retries + 1}/{self.max_retries} after {wait_time:.1f}s...")
            time.sleep(wait_time)
            
            # Mark for retry
            context.metadata['retries'] = retries + 1
            context.metadata['should_retry'] = True
        else:
            print(f"❌ Max retries ({self.max_retries}) exceeded")
            context.metadata['should_retry'] = False
