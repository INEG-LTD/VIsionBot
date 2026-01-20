"""
Browser - A powerful, vision-based web automation framework.

This package provides intelligent web automation using AI vision models to interact
with web pages like a human would. It combines computer vision, large language models,
and Playwright to create autonomous agents that can understand and interact with any web interface.

Main Classes:
    Browser: Main bot class for web automation
    Config: Configuration for the bot
    BrowserProvider: Browser management abstraction

Example:
    >>> from browser_vision_bot import Browser, Config
    >>> from browser_vision_bot import create_browser_provider
    >>> 
    >>> config = Config()
    >>> browser_provider = create_browser_provider(config.browser)
    >>> 
    >>> with Browser(config=config, browser_provider=browser_provider) as bot:
    ...     bot.page.goto("https://example.com")
    ...     bot.act("Click the 'Get Started' button")
"""

# Main bot class
from core.browser import Browser

# Configuration
from core.config import (
    Config,
    ModelConfig,
    ExecutionConfig,
    CacheConfig,
    ErrorHandlingConfig,
    BrowserConfig as BotBrowserConfig,
)

# Browser provider
from browser.provider import (
    BrowserProvider,
    LocalPlaywrightProvider,
    create_browser_provider,
    BrowserConfig,
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


# Action ledger and queue
from execution.ledger import ActionLedger, ActionStatus, ActionRecord
from execution.queue import ActionQueue

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
    "CacheConfig",
    "ErrorHandlingConfig",
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
