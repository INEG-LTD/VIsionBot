from middlewares.error_handling_middleware import ErrorHandlingMiddleware
from pathlib import Path
from browser_provider import BrowserConfig
from bot_config import BotConfig, ModelConfig, ExecutionConfig, RecordingConfig, ElementConfig, DebugConfig
from ai_utils import ReasoningLevel
from vision_bot import BrowserVisionBot
from utils.event_logger import EventType

# Example: Custom callback for normal mode (no debug prints)
def custom_event_callback(event):
    """Custom callback to handle events in normal mode"""
    # Only show important events
    if event.level in ["ERROR", "SUCCESS"]:
        print(f"[{event.level}] {event.message}")
    elif event.event_type == EventType.AGENT_ITERATION:
        print(f"🔄 {event.message}")

# Ensure the user data directory exists
user_data_path = Path.home() / "Desktop" / "bot_user_data_dir"
user_data_path.mkdir(parents=True, exist_ok=True)

# Create configuration using the new BotConfig API
config = BotConfig(
    model=ModelConfig(
        agent_model="gpt-5-nano",
        command_model="groq/meta-llama/llama-4-maverick-17b-128e-instruct",
        reasoning_level=ReasoningLevel.NONE
    ),
    execution=ExecutionConfig(
        max_attempts=30
    ),
    elements=ElementConfig(
        element_selection_fallback_model="gemini/gemini-2.5-flash-lite",
        selection_retry_attempts=2,
        overlay_only_planning=True
    ),
    recording=RecordingConfig(
        save_gif=True
    ),
    logging=DebugConfig(
        debug_mode=True  # Set to False to use callbacks only
    ),
    browser=BrowserConfig(
        provider_type="persistent",
        headless=False,
        apply_stealth=False,
        user_data_dir=str(user_data_path)
    )
)

# Create bot with config
bot = BrowserVisionBot(config=config)

# Optional: Register custom callback (works in both modes)
bot.event_logger.register_callback(custom_event_callback)
bot.use(ErrorHandlingMiddleware())
bot.start()
bot.page.goto("https://google.com")

# Run agentic mode - now returns AgentResult with extracted data
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

# Check if task succeeded
if result.success:
    print(f"✅ Task completed! Confidence: {result.confidence:.2f}")
    print(f"Reasoning: {result.reasoning}")
    
    # Access extracted data if any
    if result.extracted_data:
        print("\n📊 Extracted Data:")
        for prompt, data in result.extracted_data.items():
            print(f"  {prompt}: {data}")
else:
    print(f"❌ Task failed: {result.reasoning}")

input("Press Enter to continue...")