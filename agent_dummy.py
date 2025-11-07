from vision_bot import BrowserVisionBot

bot = BrowserVisionBot()
bot.start()
bot.page.goto("https:/google.com/")

# Run agentic mode - now returns AgentResult with extracted data
# bot.act("defer")
result = bot.agentic_mode(
    "i need to write a report on elon musk in google docs and i want to use the information about him from wikipedia, his company spacex, and also how he is doing in the stock market",
    base_knowledge=[
        "just press enter after you've typed a search term into a search field",
        "if asked to search use the best search box contextually available",
        "if you encounter a captcha, give control to the user"
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