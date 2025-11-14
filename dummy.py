from ai_utils import ReasoningLevel
from vision_bot import BrowserVisionBot

bot = BrowserVisionBot(
    save_gif=True,
    model_name="groq/meta-llama/llama-4-scout-17b-16e-instruct",
    reasoning_level=ReasoningLevel.NONE,
    fast_mode=True,
)
bot.start()
bot.goto("https://google.com")
bot.default_interpretation_mode = "semantic"

# bot.act("type: 'chris' into the name input field", max_retries=1)
# bot.act("type: 'password' into the password input field", max_retries=1)
# bot.act("click: 'coffee' in the favorite drink checkbox field", max_retries=1)
# bot.act("click: 'blue' as my favorite color", max_retries=1)
# bot.act("select: select 'yes' in the 'do you like automation?' select field", max_retries=1)
bot.act("click: the accept all cookies button", max_retries=1)
bot.act("defer: 2")
bot.act("type: 'elon musk' into the search input field", max_retries=1)
bot.act("press: enter")
bot.act("scroll: down")
bot.act("scroll: down")
bot.act("scroll: down")
bot.act("scroll: down")
bot.act("defer: 2")
bot.act("click: elon musk wikipedia page link")
# bot.act("click: the store button")
# bot.act("defer: 3")
# bot.act("click: the accessories button")
# bot.act("click: the button to browse the accessories page")
# bot.act("type: 'testing my bot' into the message input field", max_retries=1)
input("Press Enter to continue...")
# bot.end()