"""
Configuration models for BrowserVisionBot.

This module provides structured, type-safe configuration using Pydantic models.
Instead of passing 30+ arguments to BrowserVisionBot, you can create a BotConfig
object with grouped settings.

Example:
    >>> from bot_config import BotConfig, ModelConfig, ExecutionConfig
    >>> config = BotConfig(
    ...     model=ModelConfig(agent_model="gpt-5-mini"),
    ...     execution=ExecutionConfig()
    ... )
    >>> bot = BrowserVisionBot(config=config)
"""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field
from ai_utils import ReasoningLevel
from browser_provider import BrowserConfig as BrowserProviderConfig

# Backwards-compat: alias for older imports
BrowserConfig = BrowserProviderConfig


class ModelConfig(BaseModel):
    """AI model configuration for planning and execution."""
    
    model_name: str = Field(
        default="gpt-5-mini",
        description="Default model for all operations"
    )
    agent_model: str = Field(
        default="gpt-5-mini",
        description="Model used for high-level agent decisions"
    )
    command_model: str = Field(
        default="gpt-5-mini",
        description="Model used for command generation"
    )
    reasoning_level: ReasoningLevel = Field(
        default=ReasoningLevel.MEDIUM,
        description="Default reasoning level for all operations"
    )
    agent_reasoning_level: ReasoningLevel = Field(
        default=ReasoningLevel.MEDIUM,
        description="Reasoning level for agent decisions"
    )
    command_reasoning_level: ReasoningLevel = Field(
        default=ReasoningLevel.MEDIUM,
        description="Reasoning level for command generation"
    )
    
    class Config:
        arbitrary_types_allowed = True


class ExecutionConfig(BaseModel):
    """Runtime execution behavior configuration."""
    
    max_attempts: int = Field(
        default=10,
        ge=1,
        description="Maximum number of attempts for task completion"
    )
    parallel_completion_and_action: bool = Field(
        default=True,
        description="Run completion check and next action in parallel"
    )
    dedup_mode: str = Field(
        default="auto",
        description="Deduplication mode: 'auto', 'on', or 'off'"
    )
    dedup_history_quantity: int = Field(
        default=-1,
        description="Number of interactions to track for dedup (-1 = unlimited)"
    )
    completion_mode: str = Field(
        default="agent_only",
        description="Task completion determination mode: 'agent_only' (agent decides via complete: command), 'hybrid' (agent can complete OR external validation can force complete), 'external_only' (legacy mode using external CompletionContract)"
    )
    enable_sub_agents: bool = Field(
        default=False,
        description="Enable sub-agent spawning for parallel task execution. When enabled, the agent can spawn sub-agents to handle subtasks in parallel."
    )

    class Config:
        arbitrary_types_allowed = True


class CacheConfig(BaseModel):
    """Plan caching configuration."""
    
    enabled: bool = Field(
        default=True,
        description="Enable plan caching"
    )
    ttl: float = Field(
        default=6.0,
        ge=0.0,
        description="Time-to-live for cached plans in seconds"
    )
    max_reuse: int = Field(
        default=1,
        ge=-1,
        description="Maximum times a plan can be reused (-1 = unlimited)"
    )
    
    class Config:
        arbitrary_types_allowed = True


class ElementConfig(BaseModel):
    """Element detection and overlay configuration."""
    
    max_detailed_elements: int = Field(
        default=400,
        ge=1,
        description="Maximum number of detailed elements to include"
    )
    include_detailed_elements: bool = Field(
        default=True,
        description="Include detailed element information in prompts"
    )
    max_coordinate_overlays: int = Field(
        default=600,
        ge=1,
        description="Maximum number of coordinate overlays"
    )
    two_pass_planning: bool = Field(
        default=True,
        description="Use two-pass planning for element selection"
    )
    merge_overlay_selection: bool = Field(
        default=True,
        description="Merge overlay selection with plan generation"
    )
    overlay_only_planning: bool = Field(
        default=False,
        description="Return only overlay index from planning"
    )
    overlay_mode: str = Field(
        default="interactive",
        description="Overlay drawing mode: 'interactive' (default) or 'all'"
    )
    show_overlays: bool = Field(
        default=False,
        description="Show visual overlays on page elements (default: False, overrides debug mode)"
    )
    include_textless_overlays: bool = Field(
        default=False,
        description="Keep overlays with no text/aria/placeholder in LLM selection lists"
    )
    overlay_selection_max_samples: Optional[int] = Field(
        default=None,
        description="Maximum samples for overlay selection"
    )
    selection_retry_attempts: int = Field(
        default=3,
        ge=1,
        description="Number of retry attempts for element selection"
    )
    selection_fallback_model: Optional[str] = Field(
        default=None,
        description="Fallback model for element selection retries"
    )
    include_overlays_in_agent_context: bool = Field(
        default=True,
        description="Include overlay element data in agent's context for action determination. When enabled, the agent receives detailed element information (tag, placeholder, text, aria-label, etc.) to create more descriptive actions. NOTE: This only affects the agent's action determination phase. Overlays are still generated for element selection during action execution, as they are required for the system to identify and interact with elements on the page."
    )
    include_visible_text_in_agent_context: bool = Field(
        default=False,
        description="Include visible text in agent's context for action determination. When enabled, the agent receives text content from the page (viewport-only, first 2000 chars). When disabled, the agent relies purely on the screenshot for visual context. Disabling this prevents the agent from targeting off-screen elements based on text hints."
    )

    class Config:
        arbitrary_types_allowed = True


class DebugConfig(BaseModel):
    """Debugging and logging configuration."""
    
    debug_mode: bool = Field(
        default=True,
        description="Enable debug mode with verbose logging"
    )
    
    class Config:
        arbitrary_types_allowed = True


class ErrorHandlingConfig(BaseModel):
    """Error handling and recovery configuration."""
    
    screenshot_on_error: bool = Field(
        default=True,
        description="Take screenshot when errors occur"
    )
    screenshot_dir: str = Field(
        default="error_screenshots",
        description="Directory for error screenshots"
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum retry attempts for recoverable errors"
    )
    retry_delay: float = Field(
        default=2.0,
        ge=0.0,
        description="Delay between retries in seconds"
    )
    retry_backoff: float = Field(
        default=2.0,
        ge=1.0,
        description="Backoff multiplier for exponential retry"
    )
    abort_on_critical: bool = Field(
        default=True,
        description="Abort automation on critical errors"
    )
    
    class Config:
        arbitrary_types_allowed = True


class ActFunctionConfig(BaseModel):
    """
    Configuration for act() function parameters used by the agent.
    
    This allows you to selectively disable certain parameters when the agent
    calls the act() function during execution.
    """
    
    enable_target_context_guard: bool = Field(
        default=True,
        description="Enable target_context_guard parameter (contextual element filtering)"
    )
    enable_modifier: bool = Field(
        default=True,
        description="Enable modifier parameter (ordinal selection, etc.)"
    )
    enable_additional_context: bool = Field(
        default=True,
        description="Enable additional_context parameter (supplementary information)"
    )
    
    class Config:
        arbitrary_types_allowed = True


class UserMessagesConfig(BaseModel):
    """Configuration for user-facing messages."""
    
    file_upload_prompt: str = Field(
        default="    ⏸️ Waiting for user to finish selecting a file. Press Enter to continue...",
        description="Message shown when waiting for user to select a file for upload"
    )
    file_upload_interrupted: str = Field(
        default="    ⚠️ Input unavailable or interrupted; continuing without confirmation.",
        description="Message shown when file upload input is interrupted or unavailable"
    )
    
    class Config:
        arbitrary_types_allowed = True


class BotConfig(BaseModel):
    """
    Main configuration object for BrowserVisionBot.
    
    This provides a structured, type-safe way to configure the bot instead of
    passing 30+ individual arguments.
    
    Example:
        >>> config = BotConfig(
        ...     model=ModelConfig(agent_model="gpt-5-mini"),
        ...     execution=ExecutionConfig()
        ... )
        >>> bot = BrowserVisionBot(config=config)
    """
    
    model: ModelConfig = Field(
        default_factory=ModelConfig,
        description="AI model configuration"
    )
    execution: ExecutionConfig = Field(
        default_factory=ExecutionConfig,
        description="Execution behavior configuration"
    )
    cache: CacheConfig = Field(
        default_factory=CacheConfig,
        description="Plan caching configuration"
    )
    elements: ElementConfig = Field(
        default_factory=ElementConfig,
        description="Element detection configuration"
    )
    logging: DebugConfig = Field(
        default_factory=DebugConfig,
        description="Debug and logging configuration"
    )
    browser: BrowserProviderConfig = Field(
        default_factory=BrowserProviderConfig,
        description="Browser provider configuration"
    )
    error_handling: ErrorHandlingConfig = Field(
        default_factory=ErrorHandlingConfig,
        description="Error handling and recovery configuration"
    )
    act_function: ActFunctionConfig = Field(
        default_factory=ActFunctionConfig,
        description="Act function parameter configuration"
    )
    user_messages: UserMessagesConfig = Field(
        default_factory=UserMessagesConfig,
        description="User-facing messages configuration"
    )
    
    class Config:
        arbitrary_types_allowed = True
    
    @classmethod
    def debug(cls) -> BotConfig:
        """
        Create a configuration optimized for debugging.
        
        Returns:
            BotConfig with debug mode enabled
        """
        return cls(
            logging=DebugConfig(debug_mode=True),
            execution=ExecutionConfig(
                parallel_completion_and_action=False  # Sequential for easier debugging
            )
        )
    
    @classmethod
    def production(cls) -> BotConfig:
        """
        Create a configuration optimized for production use.
        
        Returns:
            BotConfig with balanced settings for reliability
        """
        return cls(
            execution=ExecutionConfig(
                max_attempts=15
            ),
            logging=DebugConfig(debug_mode=False)
        )
    
    @classmethod
    def minimal(cls) -> BotConfig:
        """
        Create a minimal configuration with defaults.
        
        Returns:
            BotConfig with all default settings
        """
        return cls()
