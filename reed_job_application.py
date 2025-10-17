from vision_bot import BrowserVisionBot

# Initialize bot with GIF recording enabled
bot = BrowserVisionBot(save_gif=True)
bot.start()
bot.goto("https://www.reed.co.uk/")
bot.default_interpretation_mode = "semantic"

def sign_in_without_password_commands(bot: BrowserVisionBot):
        bot.register_prompts([
                "form: type 'eromoseleinegbe@gmail.com' into: the email input field (look for a input field like 'Email' or 'Email Address' in the page.",
                "click: the sign in without password button",
                "defer"
        ], "sign_in_without_password_commands", all_must_be_true=False)
        
        # Search for ios developer jobs in london or remote
        bot.act("form: type 'ios developer' into the what field and 'london or remote' into the where field")
        bot.act("click: the search jobs button")

        # If user is not logged in, click the button to login and sign in without password
        bot.act("if: a sign in button/link/text is visible (look for a button like 'Sign in' or 'Login' in the page.) then: click: the sign in button (look for a button like 'Sign in' or 'Login' in the page.)")
        bot.act("if: on sign in page then: ref: sign_in_without_password_commands")

bot.register_prompts([
        "dedup: enable",
        "click: an ios job listing button/link (ignore links like 'ios jobs' or 'ios developer jobs' or 'ios developer jobs in london' or 'ios developer jobs in remote')",
        "dedup: disable",
        "click: the apply button (look for a button like 'Apply' or 'Apply Now' in the job listing. it could also look like an Easy Apply button, those are also valid)",
        "click: the button to submit the application (look for a button like 'Submit' or 'Submit Application' in modal)",
        "press: escape"
], "click_ios_job_action",
        additional_context="look for text surrounding a list of relevant job listings that are relevant to the search, e.g 'IOS Developer', 'Senior IOS Developer', 'Junior IOS Developer', etc",
        target_context_guard="only click a job listing if it its card does NOT have an 'Applied' status or an applied tag",
        confirm_before_interaction=True,
        all_must_be_true=False,
        command_id="click_ios_job_action")

# If cookie banner is visible, click the button to accept cookies
# bot.on_new_page_load(["if: a cookie banner/dialog is visible then: click: the button to accept cookies (look for a button like 'Accept' or 'Accept all' in the cookie banner)"], command_id="cookie_banner_commands")
# sign_in_without_password_commands(bot)

bot.act("defer")

# Apply to all available jobs, then click pagination
# bot.act("while: not at bottom of the page do: ref: click_ios_job_action")

# # Scroll to the top of the page
# bot.act("scroll: to top")
bot.act("while: no pagination control is visible do: scroll: down 200px", additional_context="The pagination control must have a 'Previous' and 'Next' button")
bot.act("click: the next button in the pagination control", additional_context="The pagination control must have a 'Previous' and 'Next' button")
# End the bot session (performs cleanup and terminates)
gif_path = bot.end()

input("Press enter to continue")