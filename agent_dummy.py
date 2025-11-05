from vision_bot import BrowserVisionBot

bot = BrowserVisionBot()
bot.start()
bot.page.goto("https://google.com/")

# bot.act("defer")
# Run agentic mode
success = bot.agentic_mode(
    "navigate to yahoo finance then search for 'figma' and press enter",
    base_knowledge=[
        "just press enter after you've typed a search term into a search field",
        "if asked to search use the best search box contextually available"
    ]
)

input("Press Enter to continue...")