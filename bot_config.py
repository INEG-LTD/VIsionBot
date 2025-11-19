"""
Configuration models for BrowserVisionBot.

This module provides structured, type-safe configuration using Pydantic models.
Instead of passing 30+ arguments to BrowserVisionBot, you can create a BotConfig
object with grouped settings.

Example:
    >>> from bot_config import BotConfig, ModelConfig, ExecutionConfig
    >>> config = BotConfig(
    ...     model=ModelConfig(agent_model="gpt-5-mini"),
    ...     execution=ExecutionConfig(fast_mode=True)
    ... )
    >>> bot = BrowserVisionBot(config=config)
"""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field
from ai_utils import ReasoningLevel
from browser_provider import BrowserConfig as BrowserProviderConfig


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
    fast_mode: bool = Field(
        default=False,
        description="Enable fast mode (direct keyword -> action execution)"
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


class StuckDetectorConfig(BaseModel):
    """Stuck detection configuration."""
    
    enabled: bool = Field(
        default=True,
        description="Enable stuck detection"
    )
    window_size: int = Field(
        default=5,
        ge=1,
        description="Number of recent actions to analyze"
    )
    threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Threshold for stuck detection (0.0-1.0)"
    )
    weight_repeated_action: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Weight for repeated action detection"
    )
    weight_repetitive_action_no_change: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Weight for repetitive actions with no state change"
    )
    weight_no_state_change: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for no state change detection"
    )
    weight_no_progress: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Weight for no progress detection"
    )
    weight_error_spiral: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Weight for error spiral detection"
    )
    weight_high_confidence_no_progress: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Weight for high confidence but no progress"
    )
    
    class Config:
        arbitrary_types_allowed = True


class RecordingConfig(BaseModel):
    """GIF recording configuration."""
    
    save_gif: bool = Field(
        default=False,
        description="Enable GIF recording of browser interactions"
    )
    output_dir: str = Field(
        default="gif_recordings",
        description="Directory for saving GIF recordings"
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


class BotConfig(BaseModel):
    """
    Main configuration object for BrowserVisionBot.
    
    This provides a structured, type-safe way to configure the bot instead of
    passing 30+ individual arguments.
    
    Example:
        >>> config = BotConfig(
        ...     model=ModelConfig(agent_model="gpt-5-mini"),
        ...     execution=ExecutionConfig(fast_mode=True),
        ...     recording=RecordingConfig(save_gif=True)
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
    stuck_detector: StuckDetectorConfig = Field(
        default_factory=StuckDetectorConfig,
        description="Stuck detection configuration"
    )
    recording: RecordingConfig = Field(
        default_factory=RecordingConfig,
        description="GIF recording configuration"
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
    
    class Config:
        arbitrary_types_allowed = True
    
    @classmethod
    def fast(cls) -> BotConfig:
        """
        Create a configuration optimized for speed.
        
        Returns:
            BotConfig with fast_mode enabled and reduced overhead
        """
        return cls(
            execution=ExecutionConfig(
                fast_mode=True,
                parallel_completion_and_action=True
            ),
            elements=ElementConfig(
                overlay_only_planning=True,
                two_pass_planning=False
            ),
            logging=DebugConfig(debug_mode=False)
        )
    
    @classmethod
    def debug(cls) -> BotConfig:
        """
        Create a configuration optimized for debugging.
        
        Returns:
            BotConfig with debug mode enabled and GIF recording
        """
        return cls(
            recording=RecordingConfig(save_gif=True),
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
                fast_mode=True,
                max_attempts=15
            ),
            stuck_detector=StuckDetectorConfig(
                enabled=True,
                threshold=0.5  # More sensitive in production
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
