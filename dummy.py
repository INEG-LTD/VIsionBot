from vision_bot import BrowserVisionBot

bot = BrowserVisionBot(save_gif=True, model_name="gpt-5-nano", reasoning_level="medium")
bot.start()
bot.goto("https://practice-automation.com/form-fields/")
bot.default_interpretation_mode = "semantic"

# bot.act("type: 'chris' into the name input field", max_retries=1)
# bot.act("type: 'password' into the password input field", max_retries=1)
# bot.act("click: 'coffee' in the favorite drink checkbox field", max_retries=1)
# bot.act("click: 'blue' as my favorite color", max_retries=1)
# bot.act("select: select 'yes' in the 'do you like automation?' select field", max_retries=1)
bot.act("type: 'email@email.com' into the email input field", max_retries=1)
# bot.act("type: 'testing my bot' into the message input field", max_retries=1)
input("Press Enter to continue...")
# bot.end()