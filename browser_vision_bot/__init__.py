"""
Public package surface for Browser.

This module re-exports the primary classes and helpers so consumers can simply:

    from browser_vision_bot import Browser, Config
"""

# Main bot
from core.browser import Browser

# Configuration
from core.config import (
    Config,
    ModelConfig,
    ExecutionConfig,
    CacheConfig,
    ErrorHandlingConfig,
    ActFunctionConfig,
)

# Browser provider
from browser.provider import (
    BrowserProvider,
    LocalPlaywrightProvider,
    create_browser_provider,
    BrowserConfig,
)

# Results
from execution.result import ActionResult
from agent.results import MissionResult

# Errors
from lib.errors import (
    BotError,
    NetworkError,
    ElementNotFoundError,
    ElementNotInteractableError,
    NavigationError,
    TimeoutError,
    CaptchaDetectedError,
    AuthenticationError,
    RateLimitError,
    ExtractionError,
    LLMError,
    CostLimitError,
    StuckDetectedError,
    ValidationError,
    ConfigurationError,
    BotNotStartedError,
    BotTerminatedError,
    ActionFailedError,
    ErrorContext,
    ErrorSeverity,
    RecoveryStrategy,
)

# Utilities
from lib.ai import ReasoningLevel
from middleware.base import MiddlewareManager, ActionContext, Middleware
from execution.ledger import ActionLedger, ActionStatus, ActionRecord
from execution.queue import ActionQueue
from utils.event_logger import EventLogger, set_event_logger

__version__ = "0.1.1"

__all__ = [
    # Main bot
    "Browser",
    # Configuration
    "Config",
    "ModelConfig",
    "ExecutionConfig",
    "CacheConfig",
    "ErrorHandlingConfig",
    "ActFunctionConfig",
    # Browser provider
    "BrowserProvider",
    "LocalPlaywrightProvider",
    "create_browser_provider",
    "BrowserConfig",
    # Results
    "ActionResult",
    "MissionResult",
    # Errors
    "BotError",
    "NetworkError",
    "ElementNotFoundError",
    "ElementNotInteractableError",
    "NavigationError",
    "TimeoutError",
    "CaptchaDetectedError",
    "AuthenticationError",
    "RateLimitError",
    "ExtractionError",
    "LLMError",
    "CostLimitError",
    "StuckDetectedError",
    "ValidationError",
    "ConfigurationError",
    "BotNotStartedError",
    "BotTerminatedError",
    "ActionFailedError",
    "ErrorContext",
    "ErrorSeverity",
    "RecoveryStrategy",
    # Utilities
    "ReasoningLevel",
    "MiddlewareManager",
    "ActionContext",
    "Middleware",
    "ActionLedger",
    "ActionStatus",
    "ActionRecord",
    "ActionQueue",
    "EventLogger",
    "set_event_logger",
]

