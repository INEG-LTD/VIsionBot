"""
Structured error handling for BrowserVisionBot.

Provides custom exception types, error context, and recovery strategies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Error recovery strategies."""
    RETRY = "retry"
    SKIP = "skip"
    ABORT = "abort"
    ASK_USER = "ask_user"
    FALLBACK = "fallback"


@dataclass
class ErrorContext:
    """
    Context information about an error.
    
    Captures everything needed to understand and debug an error.
    """
    
    error_type: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Browser state
    page_url: Optional[str] = None
    page_title: Optional[str] = None
    screenshot_path: Optional[str] = None
    
    # Action context
    action_type: Optional[str] = None
    action_data: Optional[Dict[str, Any]] = None
    
    # Stack trace
    traceback: Optional[str] = None
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'error_type': self.error_type,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'page_url': self.page_url,
            'page_title': self.page_title,
            'screenshot_path': self.screenshot_path,
            'action_type': self.action_type,
            'action_data': self.action_data,
            'traceback': self.traceback,
            'metadata': self.metadata
        }


class BotError(Exception):
    """
    Base exception for all bot errors.
    
    All custom exceptions should inherit from this.
    """
    
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY
    
    def __init__(
        self,
        message: str,
        context: Optional[ErrorContext] = None,
        **kwargs
    ):
        super().__init__(message)
        self.message = message
        self.context = context or ErrorContext(
            error_type=self.__class__.__name__,
            message=message
        )
        
        # Allow overriding context fields
        for key, value in kwargs.items():
            if hasattr(self.context, key):
                setattr(self.context, key, value)


class NetworkError(BotError):
    """Network-related errors (timeouts, connection failures, etc.)."""
    severity = ErrorSeverity.HIGH
    recovery_strategy = RecoveryStrategy.RETRY


class ElementNotFoundError(BotError):
    """Element could not be found on the page."""
    severity = ErrorSeverity.MEDIUM
    recovery_strategy = RecoveryStrategy.RETRY


class ElementNotInteractableError(BotError):
    """Element exists but cannot be interacted with."""
    severity = ErrorSeverity.MEDIUM
    recovery_strategy = RecoveryStrategy.RETRY


class NavigationError(BotError):
    """Page navigation failed."""
    severity = ErrorSeverity.HIGH
    recovery_strategy = RecoveryStrategy.RETRY


class TimeoutError(BotError):
    """Operation timed out."""
    severity = ErrorSeverity.MEDIUM
    recovery_strategy = RecoveryStrategy.RETRY


class CaptchaDetectedError(BotError):
    """CAPTCHA detected on the page."""
    severity = ErrorSeverity.HIGH
    recovery_strategy = RecoveryStrategy.ASK_USER


class AuthenticationError(BotError):
    """Authentication failed."""
    severity = ErrorSeverity.CRITICAL
    recovery_strategy = RecoveryStrategy.ABORT


class RateLimitError(BotError):
    """Rate limit exceeded."""
    severity = ErrorSeverity.HIGH
    recovery_strategy = RecoveryStrategy.RETRY


class ExtractionError(BotError):
    """Data extraction failed."""
    severity = ErrorSeverity.MEDIUM
    recovery_strategy = RecoveryStrategy.RETRY


class LLMError(BotError):
    """LLM API call failed."""
    severity = ErrorSeverity.HIGH
    recovery_strategy = RecoveryStrategy.RETRY


class CostLimitError(BotError):
    """Cost limit exceeded."""
    severity = ErrorSeverity.CRITICAL
    recovery_strategy = RecoveryStrategy.ABORT


class StuckDetectedError(BotError):
    """Bot appears to be stuck in a loop."""
    severity = ErrorSeverity.HIGH
    recovery_strategy = RecoveryStrategy.FALLBACK


class ValidationError(BotError):
    """Data validation failed."""
    severity = ErrorSeverity.MEDIUM
    recovery_strategy = RecoveryStrategy.SKIP


class ConfigurationError(BotError):
    """Invalid configuration."""
    severity = ErrorSeverity.CRITICAL
    recovery_strategy = RecoveryStrategy.ABORT


@dataclass
class ErrorHandler:
    """
    Handles errors with configurable recovery strategies.
    """
    
    # Configuration
    screenshot_on_error: bool = True
    screenshot_dir: str = "error_screenshots"
    max_retries: int = 3
    retry_delay: float = 2.0
    
    # Error history
    errors: List[ErrorContext] = field(default_factory=list)
    
    def handle_error(
        self,
        error: Exception,
        bot: Any,
        action_context: Optional[Dict[str, Any]] = None
    ) -> RecoveryStrategy:
        """
        Handle an error and determine recovery strategy.
        
        Args:
            error: The exception that occurred
            bot: Reference to the bot instance
            action_context: Context about the action that failed
            
        Returns:
            RecoveryStrategy to use
        """
        # Create error context
        if isinstance(error, BotError):
            context = error.context
        else:
            context = ErrorContext(
                error_type=type(error).__name__,
                message=str(error)
            )
        
        # Capture browser state
        if bot and hasattr(bot, 'page') and bot.page:
            try:
                context.page_url = bot.page.url
                context.page_title = bot.page.title()
                
                # Take screenshot if enabled
                if self.screenshot_on_error:
                    import os
                    os.makedirs(self.screenshot_dir, exist_ok=True)
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{context.error_type}_{timestamp}.png"
                    screenshot_path = os.path.join(self.screenshot_dir, filename)
                    
                    bot.page.screenshot(path=screenshot_path)
                    context.screenshot_path = screenshot_path
            except Exception:
                pass  # Don't fail if we can't capture state
        
        # Add action context
        if action_context:
            context.action_type = action_context.get('action_type')
            context.action_data = action_context.get('action_data')
        
        # Store error
        self.errors.append(context)
        
        # Determine recovery strategy
        if isinstance(error, BotError):
            return error.recovery_strategy
        else:
            return RecoveryStrategy.RETRY
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all errors."""
        error_counts = {}
        for error in self.errors:
            error_type = error.error_type
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        return {
            'total_errors': len(self.errors),
            'error_counts': error_counts,
            'recent_errors': [e.to_dict() for e in self.errors[-5:]]
        }
    
    def clear_errors(self) -> None:
        """Clear error history."""
        self.errors.clear()
