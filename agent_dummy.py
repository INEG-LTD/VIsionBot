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

# Create bot with debug_mode=True (default) to see all prints
# Set debug_mode=False for normal mode (only callbacks, no prints)
bot = BrowserVisionBot(
    save_gif=True,
    agent_model_name="gpt-5-mini",
    command_model_name="groq/meta-llama/llama-4-maverick-17b-128e-instruct",
    element_selection_fallback_model="gemini/gemini-2.5-flash-lite",
    element_selection_retry_attempts=2,
    reasoning_level=ReasoningLevel.NONE,
    overlay_only_planning=True,
    fast_mode=True,
    debug_mode=False,  # Set to False to use callbacks only
)

# Optional: Register custom callback (works in both modes)
bot.event_logger.register_callback(custom_event_callback)
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