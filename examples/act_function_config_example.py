"""
Example: Controlling Act Function Parameters

This example demonstrates how to selectively enable/disable parameters
that the agent uses when calling the act() function during autonomous execution.

Use cases:
- Disable target_context_guard: When the agent is too restrictive in element selection
- Disable modifier: When you don't want ordinal selection (first, second, etc.)
- Disable additional_context: When you want minimal context for faster execution
"""

import sys
from pathlib import Path

# Add parent directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from vision_bot import BrowserVisionBot
from bot_config import (
    BotConfig,
    ModelConfig,
    ExecutionConfig,
    ActFunctionConfig,
)
from browser_provider import create_browser_provider
from ai_utils import ReasoningLevel


def example_default_config():
    """Example with default settings - all act parameters enabled"""
    print("\n=== Example 1: Default Configuration (All Parameters Enabled) ===\n")
    
    config = BotConfig(
        model=ModelConfig(
            agent_model="gpt-5-mini",
            reasoning_level=ReasoningLevel.NONE
        ),
        execution=ExecutionConfig(max_attempts=10),
        # Default ActFunctionConfig - all parameters enabled
        act_function=ActFunctionConfig(
            enable_target_context_guard=True,  # Contextual filtering enabled
            enable_modifier=True,               # Ordinal selection enabled
            enable_additional_context=True      # Extra context enabled
        )
    )
    
    print(f"✓ target_context_guard: Enabled")
    print(f"✓ modifier: Enabled")
    print(f"✓ additional_context: Enabled")
    print("\nAgent will use all available parameters for precise element targeting.")


def example_disable_target_context_guard():
    """Example: Disable target_context_guard for simpler element targeting"""
    print("\n=== Example 2: Disable target_context_guard ===\n")
    
    config = BotConfig(
        model=ModelConfig(
            agent_model="gpt-5-mini",
            reasoning_level=ReasoningLevel.NONE
        ),
        execution=ExecutionConfig(max_attempts=10),
        # Disable target_context_guard for less restrictive targeting
        act_function=ActFunctionConfig(
            enable_target_context_guard=False,  # Disable contextual filtering
            enable_modifier=True,
            enable_additional_context=True
        )
    )
    
    print(f"✗ target_context_guard: Disabled")
    print(f"✓ modifier: Enabled")
    print(f"✓ additional_context: Enabled")
    print("\nUse case: When the agent is too restrictive in element selection")
    print("or when you want simpler, more straightforward element targeting.")


def example_disable_ordinal_selection():
    """Example: Disable modifier to avoid ordinal selection"""
    print("\n=== Example 3: Disable Ordinal Selection ===\n")
    
    config = BotConfig(
        model=ModelConfig(
            agent_model="gpt-5-mini",
            reasoning_level=ReasoningLevel.NONE
        ),
        execution=ExecutionConfig(max_attempts=10),
        # Disable modifier to avoid ordinal (first, second, etc.) selection
        act_function=ActFunctionConfig(
            enable_target_context_guard=True,
            enable_modifier=False,              # Disable ordinal selection
            enable_additional_context=True
        )
    )
    
    print(f"✓ target_context_guard: Enabled")
    print(f"✗ modifier: Disabled")
    print(f"✓ additional_context: Enabled")
    print("\nUse case: When ordinal selection causes confusion or when you want")
    print("the agent to select elements without positional constraints.")


def example_minimal_config():
    """Example: Minimal configuration with all extras disabled"""
    print("\n=== Example 4: Minimal Configuration ===\n")
    
    config = BotConfig(
        model=ModelConfig(
            agent_model="gpt-5-mini",
            reasoning_level=ReasoningLevel.NONE
        ),
        execution=ExecutionConfig(max_attempts=10),
        # Disable all extra parameters for simplest targeting
        act_function=ActFunctionConfig(
            enable_target_context_guard=False,
            enable_modifier=False,
            enable_additional_context=False
        )
    )
    
    print(f"✗ target_context_guard: Disabled")
    print(f"✗ modifier: Disabled")
    print(f"✗ additional_context: Disabled")
    print("\nUse case: When you want the simplest possible element targeting")
    print("without any contextual filtering or additional information.")


def example_with_actual_bot():
    """
    Real example showing how to use the configuration with an actual bot.
    This example would work if you have a browser running.
    """
    print("\n=== Example 5: Using Config with Actual Bot ===\n")
    
    # Create config with target_context_guard disabled
    config = BotConfig(
        model=ModelConfig(
            agent_model="gpt-5-mini",
            reasoning_level=ReasoningLevel.NONE
        ),
        execution=ExecutionConfig(max_attempts=10),
        act_function=ActFunctionConfig(
            enable_target_context_guard=False,  # Simpler targeting
            enable_modifier=True,
            enable_additional_context=True
        )
    )
    
    print("Configuration created with target_context_guard disabled.")
    print("\nTo use this configuration:")
    print("""
    # Create browser provider
    browser_provider = create_browser_provider(config.browser)
    
    # Create bot with config
    bot = BrowserVisionBot(config=config, browser_provider=browser_provider)
    bot.start()
    
    # Navigate and execute task
    bot.page.goto("https://example.com")
    result = bot.execute_task(
        "Fill out the form",
        base_knowledge=["use reasonable defaults if values not provided"]
    )
    
    # The agent will now use simpler element targeting without
    # contextual guard constraints
    """)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("Act Function Configuration Examples")
    print("="*70)
    
    # Run all examples
    example_default_config()
    example_disable_target_context_guard()
    example_disable_ordinal_selection()
    example_minimal_config()
    example_with_actual_bot()
    
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print("""
The ActFunctionConfig allows you to fine-tune how the agent calls the 
act() function during autonomous execution:

• enable_target_context_guard: Controls contextual element filtering
  - True (default): More precise but potentially more restrictive
  - False: Simpler, more direct element targeting

• enable_modifier: Controls ordinal selection (first, second, etc.)
  - True (default): Allows positional selection
  - False: Disables positional constraints

• enable_additional_context: Controls supplementary planning information
  - True (default): Provides extra context for planning
  - False: Minimal context for faster execution

Choose the configuration that best fits your automation needs!
    """)




