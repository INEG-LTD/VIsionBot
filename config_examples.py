"""
Examples demonstrating the BotConfig API.

This file shows various ways to configure BrowserVisionBot using the
configuration object pattern.
"""

from bot_config import BotConfig, ModelConfig, ExecutionConfig, RecordingConfig, StuckDetectorConfig
from vision_bot import BrowserVisionBot
from ai_utils import ReasoningLevel


def example_1_basic_usage():
    """Example 1: Basic usage with default configuration"""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    # Create bot with all defaults
    config = BotConfig()
    bot = BrowserVisionBot(config=config)
    
    print("✓ Created bot with default configuration")
    print(f"  - Model: {bot.model_name}")
    print(f"  - Fast mode: {bot.fast_mode}")
    print(f"  - Debug mode: {config.logging.debug_mode}")
    print()


def example_2_custom_configuration():
    """Example 2: Custom configuration with specific settings"""
    print("=" * 60)
    print("Example 2: Custom Configuration")
    print("=" * 60)
    
    # Create custom configuration
    config = BotConfig(
        model=ModelConfig(
            agent_model="gpt-5-mini",
            command_model="groq/meta-llama/llama-4-maverick-17b-128e-instruct",
            reasoning_level=ReasoningLevel.NONE
        ),
        execution=ExecutionConfig(
            fast_mode=True,
            max_attempts=15
        ),
        recording=RecordingConfig(
            save_gif=True
        )
    )
    
    bot = BrowserVisionBot(config=config)
    
    print("✓ Created bot with custom configuration")
    print(f"  - Agent model: {bot.agent_model_name}")
    print(f"  - Command model: {bot.command_model_name}")
    print(f"  - Fast mode: {bot.fast_mode}")
    print(f"  - Max attempts: {bot.max_attempts}")
    print(f"  - GIF recording: {bot.save_gif}")
    print()


def example_3_preset_configurations():
    """Example 3: Using preset configurations"""
    print("=" * 60)
    print("Example 3: Preset Configurations")
    print("=" * 60)
    
    # Fast configuration - optimized for speed
    fast_config = BotConfig.fast()
    fast_bot = BrowserVisionBot(config=fast_config)
    print("✓ Fast preset:")
    print(f"  - Fast mode: {fast_bot.fast_mode}")
    print(f"  - Debug mode: {fast_config.logging.debug_mode}")
    
    # Debug configuration - optimized for debugging
    debug_config = BotConfig.debug()
    debug_bot = BrowserVisionBot(config=debug_config)
    print("✓ Debug preset:")
    print(f"  - GIF recording: {debug_bot.save_gif}")
    print(f"  - Debug mode: {debug_config.logging.debug_mode}")
    print(f"  - Parallel execution: {debug_bot.parallel_completion_and_action}")
    
    # Production configuration - optimized for reliability
    prod_config = BotConfig.production()
    prod_bot = BrowserVisionBot(config=prod_config)
    print("✓ Production preset:")
    print(f"  - Fast mode: {prod_bot.fast_mode}")
    print(f"  - Max attempts: {prod_bot.max_attempts}")
    print(f"  - Stuck detector threshold: {prod_bot.stuck_detector_threshold}")
    print()


def example_4_modifying_presets():
    """Example 4: Modifying preset configurations"""
    print("=" * 60)
    print("Example 4: Modifying Presets")
    print("=" * 60)
    
    # Start with fast preset and modify specific settings
    config = BotConfig.fast()
    config.recording.save_gif = True  # Enable GIF recording
    config.execution.max_attempts = 20  # Increase max attempts
    
    bot = BrowserVisionBot(config=config)
    
    print("✓ Created bot with modified fast preset")
    print(f"  - Fast mode (from preset): {bot.fast_mode}")
    print(f"  - GIF recording (modified): {bot.save_gif}")
    print(f"  - Max attempts (modified): {bot.max_attempts}")
    print()


def example_5_all_settings():
    """Example 5: Comprehensive configuration with all settings"""
    print("=" * 60)
    print("Example 5: All Settings")
    print("=" * 60)
    
    config = BotConfig(
        model=ModelConfig(
            agent_model="gpt-5-mini",
            command_model="groq/meta-llama/llama-4-maverick-17b-128e-instruct",
            reasoning_level=ReasoningLevel.NONE,
            agent_reasoning_level=ReasoningLevel.LOW,
            command_reasoning_level=ReasoningLevel.NONE
        ),
        execution=ExecutionConfig(
            fast_mode=True,
            max_attempts=20,
            parallel_completion_and_action=True,
            dedup_mode="auto",
            dedup_history_quantity=-1
        ),
        recording=RecordingConfig(
            save_gif=True,
            output_dir="my_recordings"
        ),
        stuck_detector=StuckDetectorConfig(
            enabled=True,
            threshold=0.5,
            window_size=7
        )
    )
    
    bot = BrowserVisionBot(config=config)
    
    print("✓ Created bot with comprehensive configuration")
    print(f"  - Agent model: {bot.agent_model_name}")
    print(f"  - Command model: {bot.command_model_name}")
    print(f"  - Fast mode: {bot.fast_mode}")
    print(f"  - Max attempts: {bot.max_attempts}")
    print(f"  - GIF recording: {bot.save_gif}")
    print(f"  - GIF output dir: {bot.gif_output_dir}")
    print(f"  - Stuck detector: {bot.stuck_detector_enabled}")
    print(f"  - Stuck threshold: {bot.stuck_detector_threshold}")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("BotConfig API Examples")
    print("=" * 60 + "\n")
    
    example_1_basic_usage()
    example_2_custom_configuration()
    example_3_preset_configurations()
    example_4_modifying_presets()
    example_5_all_settings()
    
    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)
