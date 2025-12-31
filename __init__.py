"""
BrowserVisionBot - A powerful, vision-based web automation framework.

This package provides intelligent web automation using AI vision models to interact
with web pages like a human would. It combines computer vision, large language models,
and Playwright to create autonomous agents that can understand and interact with any web interface.

Main Classes:
    BrowserVisionBot: Main bot class for web automation
    BotConfig: Configuration for the bot
    BrowserProvider: Browser management abstraction

Example:
    >>> from browser_vision_bot import BrowserVisionBot, BotConfig
    >>> from browser_vision_bot import create_browser_provider
    >>> 
    >>> config = BotConfig()
    >>> browser_provider = create_browser_provider(config.browser)
    >>> 
    >>> with BrowserVisionBot(config=config, browser_provider=browser_provider) as bot:
    ...     bot.page.goto("https://example.com")
    ...     bot.act("Click the 'Get Started' button")
"""

# Main bot class
from vision_bot import BrowserVisionBot

# Configuration
from bot_config import (
    BotConfig,
    ModelConfig,
    ExecutionConfig,
    CacheConfig,
    ErrorHandlingConfig,
    BrowserConfig as BotBrowserConfig,
)

# Browser provider
from browser_provider import (
    BrowserProvider,
    LocalPlaywrightProvider,
    create_browser_provider,
    BrowserConfig,
)

# Result types
from action_result import ActionResult
from agent.agent_result import AgentResult

# Error handling
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

# AI utilities
from ai_utils import ReasoningLevel

# Middleware
from middleware import MiddlewareManager, ActionContext, Middleware

# Action ledger and queue
from action_ledger import ActionLedger, ActionStatus, ActionRecord
from action_queue import ActionQueue

# Event logger
from utils.event_logger import EventLogger, set_event_logger

__version__ = "0.1.0"
__all__ = [
    # Main classes
    "BrowserVisionBot",
    # Configuration
    "BotConfig",
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
    # Middleware
    "MiddlewareManager",
    "ActionContext",
    "Middleware",
    # Action tracking
    "ActionLedger",
    "ActionStatus",
    "ActionRecord",
    "ActionQueue",
    # Event logging
    "EventLogger",
    "set_event_logger",
]
