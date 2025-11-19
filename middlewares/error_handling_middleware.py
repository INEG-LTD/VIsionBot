"""Error handling middleware for BrowserVisionBot."""

import time
from middleware import Middleware, ActionContext
from error_handling import (
    ErrorHandler,
    RecoveryStrategy,
    BotError,
    ErrorSeverity
)
from typing import Any


class ErrorHandlingMiddleware(Middleware):
    """
    Middleware that handles errors with configurable recovery strategies.
    
    Features:
    - Automatic screenshots on errors
    - Configurable retry logic
    - Error context preservation
    - Recovery strategy execution
    
    Example:
        >>> error_config = ErrorHandlingConfig(
        ...     screenshot_on_error=True,
        ...     max_retries=3,
        ...     retry_delay=2.0
        ... )
        >>> bot.use(ErrorHandlingMiddleware(error_config))
    """
    
    def __init__(self, config=None):
        """
        Initialize error handling middleware.
        
        Args:
            config: ErrorHandlingConfig instance
        """
        from bot_config import ErrorHandlingConfig
        
        if config is None:
            config = ErrorHandlingConfig()
        
        self.config = config
        self.error_handler = ErrorHandler(
            screenshot_on_error=config.screenshot_on_error,
            screenshot_dir=config.screenshot_dir,
            max_retries=config.max_retries,
            retry_delay=config.retry_delay
        )
    
    def on_error(self, context: ActionContext, error: Exception) -> None:
        """
        Handle error and determine recovery strategy.
        
        Args:
            context: Action context
            error: The exception that occurred
        """
        # Get recovery strategy
        strategy = self.error_handler.handle_error(
            error,
            context.bot,
            {
                'action_type': context.action_type,
                'action_data': context.action_data
            }
        )
        
        # Log error
        error_context = self.error_handler.errors[-1] if self.error_handler.errors else None
        if error_context:
            print(f"\n❌ Error: {error_context.error_type}")
            print(f"   Message: {error_context.message}")
            if error_context.page_url:
                print(f"   Page: {error_context.page_url}")
            if error_context.screenshot_path:
                print(f"   Screenshot: {error_context.screenshot_path}")
        
        # Execute recovery strategy
        if strategy == RecoveryStrategy.RETRY:
            retries = context.metadata.get('error_retries', 0)
            
            if retries < self.config.max_retries:
                # Calculate backoff delay
                delay = self.config.retry_delay * (self.config.retry_backoff ** retries)
                
                print(f"   Strategy: Retry ({retries + 1}/{self.config.max_retries})")
                print(f"   Waiting {delay:.1f}s before retry...")
                
                time.sleep(delay)
                
                # Mark for retry
                context.metadata['error_retries'] = retries + 1
                context.metadata['should_retry'] = True
            else:
                print(f"   Strategy: Max retries exceeded, aborting")
                context.metadata['should_retry'] = False
        
        elif strategy == RecoveryStrategy.ABORT:
            print(f"   Strategy: Abort (critical error)")
            context.metadata['should_retry'] = False
            
            if self.config.abort_on_critical:
                raise error
        
        elif strategy == RecoveryStrategy.SKIP:
            print(f"   Strategy: Skip action and continue")
            context.metadata['should_retry'] = False
            context.should_continue = False  # Skip this action
        
        elif strategy == RecoveryStrategy.ASK_USER:
            print(f"   Strategy: Ask user for intervention")
            try:
                response = input("   How should we proceed? (retry/skip/abort): ").lower()
                if response == 'retry':
                    context.metadata['should_retry'] = True
                elif response == 'skip':
                    context.metadata['should_retry'] = False
                    context.should_continue = False
                else:
                    raise error
            except (EOFError, KeyboardInterrupt):
                raise error
    
    def get_error_summary(self):
        """Get summary of all errors encountered."""
        return self.error_handler.get_error_summary()
    
    def clear_errors(self):
        """Clear error history."""
        self.error_handler.clear_errors()
