"""
Enhanced agent_dummy.py with middleware integration.

Demonstrates how to use middlewares for logging, metrics, and cost tracking
in a real automation scenario.
"""

from browser_provider import BrowserConfig
from bot_config import BotConfig, ModelConfig, ExecutionConfig, RecordingConfig, ElementConfig, DebugConfig
from ai_utils import ReasoningLevel
from vision_bot import BrowserVisionBot
from utils.event_logger import EventType

# Import middlewares
from middlewares import (
    LoggingMiddleware,
    MetricsMiddleware,
    CostTrackingMiddleware,
    HumanInTheLoopMiddleware
)

# Create configuration
config = BotConfig(
    model=ModelConfig(
        agent_model="gpt-5-mini",
        command_model="groq/meta-llama/llama-4-maverick-17b-128e-instruct",
        reasoning_level=ReasoningLevel.NONE
    ),
    execution=ExecutionConfig(
        fast_mode=True,
        max_attempts=30
    ),
    elements=ElementConfig(
        selection_fallback_model="gemini/gemini-2.5-flash-lite",
        selection_retry_attempts=2,
        overlay_only_planning=True
    ),
    recording=RecordingConfig(
        save_gif=True
    ),
    logging=DebugConfig(
        debug_mode=False  # Middleware will handle logging
    ),
    browser=BrowserConfig(
        provider_type="persistent",
        headless=False,
        user_data_dir="/Users/chris_ineg/Library/Application Support/Google/Chrome/Default"
    )
)

# Create bot
bot = BrowserVisionBot(config=config)

# Add middlewares (chained)
metrics = MetricsMiddleware()

bot.use(LoggingMiddleware(verbose=False)) \
   .use(metrics) \
   .use(CostTrackingMiddleware(max_cost=0.50)) \
   .use(HumanInTheLoopMiddleware(on_captcha=True))

print("=" * 60)
print("🤖 Enhanced Bot with Middleware")
print("=" * 60)
print("Middlewares active:")
print("  ✓ Logging (non-verbose)")
print("  ✓ Metrics collection")
print("  ✓ Cost tracking ($0.50 limit)")
print("  ✓ Human-in-the-loop (CAPTCHA)")
print("=" * 60)
print()

# Start bot
bot.start()
bot.page.goto("https://google.com")

# Run agentic mode - middlewares will automatically track everything
result = bot.agentic_mode(
    "go to reed job website and search for it jobs in london",
    base_knowledge=[
        "just press enter after you've typed a search term into a search field",
        "if asked to search use the best search box contextually available",
        "if you encounter a captcha, give control to the user",
        "if there is a cookie banner, accept all cookies",
    ],
    strict_mode=True
)

# Check result
if result.success:
    print(f"\n✅ Task completed! Confidence: {result.confidence:.2f}")
    print(f"Reasoning: {result.reasoning}")
    
    # Access extracted data if any
    if result.extracted_data:
        print("\n📊 Extracted Data:")
        for prompt, data in result.extracted_data.items():
            print(f"  {prompt}: {data}")
else:
    print(f"\n❌ Task failed: {result.reasoning}")

# Print metrics summary
print()
metrics.print_summary()

input("\nPress Enter to exit...")
