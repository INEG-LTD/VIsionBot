from vision_bot import BrowserVisionBot

bot = BrowserVisionBot(save_gif=True)
bot.start()
bot.goto("https://the-internet.herokuapp.com/checkboxes")
bot.default_interpretation_mode = "semantic"

bot.act("form: tick checkbox 1")
# bot.end()