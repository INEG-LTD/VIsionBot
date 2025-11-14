from ai_utils import ReasoningLevel
from vision_bot import BrowserVisionBot

bot = BrowserVisionBot(
    save_gif=True,
    agent_model_name="gpt-5-mini",
    # command_model_name="groq/meta-llama/llama-4-maverick-17b-128e-instruct",
    reasoning_level=ReasoningLevel.LOW,
    overlay_only_planning=True,
    fast_mode=True
)
bot.start()
bot.page.goto("https://google.com")

# Run agentic mode - now returns AgentResult with extracted data
result = bot.agentic_mode(
    "go to elon musk's wikipedia page",
    base_knowledge=[
        "just press enter after you've typed a search term into a search field",
        "if asked to search use the best search box contextually available",
        "if you encounter a captcha, give control to the user",
        "if there is a cookie banner, accept all cookies",
    ],
    allow_partial_completion=True
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