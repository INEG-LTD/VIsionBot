"""
Configuration models for Agent.

This module provides structured, type-safe configuration using Pydantic models.
Instead of passing 30+ arguments to Agent, you can create a Config
object with grouped settings.

"""
from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
from lib.ai import ReasoningLevel
from browser.provider import BrowserConfig as BrowserProviderConfig

# Backwards-compat: alias for older imports
BrowserConfig = BrowserProviderConfig


class TaskExecutionConfig(BaseModel):
    """Configuration for unified task execution."""

    max_actions_per_task: int = Field(
        default=200,
        ge=1,
        description="Maximum number of actions before a task is forced to end"
    )
    max_tasks_per_mission: int = Field(
        default=20,
        ge=1,
        description="Maximum number of tasks the planner can create per mission (safety limit)"
    )

    class Config:
        arbitrary_types_allowed = True


class ModelConfig(BaseModel):
    """AI model configuration for planning and execution."""

    agent_model: str = Field(
        default="gpt-5-mini",
        description="Model used for high-level agent decisions"
    )
    command_model: str = Field(
        default="gpt-5-mini",
        description="Model used for command generation"
    )
    agent_reasoning_level: ReasoningLevel = Field(
        default=ReasoningLevel.MEDIUM,
        description="Reasoning level for agent decisions"
    )
    command_reasoning_level: ReasoningLevel = Field(
        default=ReasoningLevel.MEDIUM,
        description="Reasoning level for command generation"
    )
    image_detail: str = Field(
        default="high",
        description="Image detail level for vision API: 'low' (faster, cheaper), 'high' (more accurate), or 'auto'"
    )
    
    class Config:
        arbitrary_types_allowed = True


class ExecutionConfig(BaseModel):
    """Runtime execution behavior configuration."""
    max_iterations: int = Field(
        default=500,
        ge=1,
        description="Maximum number of iterations for task completion"
    )
    auto_complete_extract_commands: bool = Field(
        default=True,
        description="Automatically mark tasks complete after successful extract: commands when the task only contains extraction actions"
    )
    max_actions_per_plan: int = Field(
        default=6,
        ge=1,
        le=20,
        description="Maximum number of actions to generate in a single action plan. Default is 6. Valid range: 1-20."
    )
    wait_for_load_before_iteration: bool = Field(
        default=False,
        description="If True, wait for the page to reach a load state before each agent iteration."
    )
    wait_for_load_state: str = Field(
        default="networkidle",
        description="Load state to wait for before each agent iteration: 'load', 'domcontentloaded', or 'networkidle'."
    )
    wait_for_load_timeout_ms: int = Field(
        default=30000,
        ge=0,
        description="Max time to wait for page load before each agent iteration (milliseconds)."
    )
    use_agent_overlay_index: bool = Field(
        default=True,
        description="If True, trust the agent's overlay_index from function calling instead of re-selecting via a separate LLM call. "
                    "Saves an API call per click/type/clear and avoids the overlay selector overriding the agent's correct choice."
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
    show_overlay_candidates: bool = Field(
        default=False,
        description="Show detailed overlay candidate information during LLM selection"
    )
    save_screenshots: bool = Field(
        default=False,
        description="Save screenshots sent to the agent for debugging"
    )
    screenshot_dir: str = Field(
        default="agent_screenshots",
        description="Directory to save agent screenshots"
    )
    show_llm_costs: bool = Field(
        default=True,
        description="Show LLM cost information in debug mode"
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


class Config(BaseModel):
    """
    Main configuration object for Agent.
    
    This provides a structured, type-safe way to configure the agent instead of
    passing 30+ individual arguments.
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
    task_execution: TaskExecutionConfig = Field(
        default_factory=TaskExecutionConfig,
        description="Unified task execution configuration"
    )

    class Config:
        arbitrary_types_allowed = True
    
    @classmethod
    def production(cls) -> Config:
        """
        Create a configuration optimized for production use.
        
        Returns:
            Config with balanced settings for reliability
        """
        return cls(
            execution=ExecutionConfig(
                max_iterations=1500
            ),
            logging=DebugConfig(debug_mode=False)
        )
    
    @classmethod
    def minimal(cls) -> Config:
        """
        Create a minimal configuration with defaults.
        
        Returns:
            Config with all default settings
        """
        return cls()
