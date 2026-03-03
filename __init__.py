
# Main bot class
from core.browser import Browser

# Configuration
from core.config import (
    Config,
    ModelConfig,
    ExecutionConfig,
)

# Browser provider
from browser.provider import (
    BrowserProvider,
    LocalPlaywrightProvider,
    create_browser_provider,
    BrowserConfig,
    BrowserConfig as BotBrowserConfig,
)

# Result types
from execution.result import ActionResult
from agent.results import AgentResult


# Error handling
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

# AI utilities
from lib.ai import ReasoningLevel

# Event logger
from utils.event_logger import EventLogger, set_event_logger

__version__ = "0.1.0"
__all__ = [
    # Main classes
    "Browser",
    # Configuration
    "Config",
    "ModelConfig",
    "ExecutionConfig",
    "BotBrowserConfig",
    # Browser provider
    "BrowserProvider",
    "LocalPlaywrightProvider",
    "create_browser_provider",
    "BrowserConfig",
    # Result types
    "ActionResult",
    "AgentResult",
    # Error handling
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
    # AI utilities
    "ReasoningLevel",
    # Action tracking
    "ActionLedger",
    "ActionStatus",
    "ActionRecord",
    "ActionQueue",
    # Event logging
    "EventLogger",
    "set_event_logger",
]
