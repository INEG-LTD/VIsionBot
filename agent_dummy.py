from vision_bot import BrowserVisionBot

bot = BrowserVisionBot()
bot.start()
bot.page.goto("https://finance.yahoo.com/quote/FIG/")

# Run agentic mode - now returns AgentResult with extracted data
# bot.act("defer")
result = bot.agentic_mode(
    "extract only the current and after market price of the stock",
    base_knowledge=[
        "just press enter after you've typed a search term into a search field",
        "if asked to search use the best search box contextually available"
    ]
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