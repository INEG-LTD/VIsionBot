"""
Public package surface for BrowserVisionBot.

This module re-exports the primary classes and helpers so consumers can simply:

    from browser_vision_bot import BrowserVisionBot, BotConfig
"""

# Main bot
from vision_bot import BrowserVisionBot

# Configuration
from bot_config import (
    BotConfig,
    ModelConfig,
    ExecutionConfig,
    CacheConfig,
    RecordingConfig,
    ErrorHandlingConfig,
    ActFunctionConfig,
)

# Browser provider
from browser_provider import (
    BrowserProvider,
    LocalPlaywrightProvider,
    create_browser_provider,
    BrowserConfig,
)

# Results
from action_result import ActionResult
from agent.agent_result import AgentResult

# Errors
from error_handling import (
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
from ai_utils import ReasoningLevel
from middleware import MiddlewareManager, ActionContext, Middleware
from action_ledger import ActionLedger, ActionStatus, ActionRecord
from action_queue import ActionQueue
from utils.event_logger import EventLogger, set_event_logger

__version__ = "0.1.1"

__all__ = [
    # Main bot
    "BrowserVisionBot",
    # Configuration
    "BotConfig",
    "ModelConfig",
    "ExecutionConfig",
    "CacheConfig",
    "RecordingConfig",
    "ErrorHandlingConfig",
    "ActFunctionConfig",
    # Browser provider
    "BrowserProvider",
    "LocalPlaywrightProvider",
    "create_browser_provider",
    "BrowserConfig",
    # Results
    "ActionResult",
    "AgentResult",
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

