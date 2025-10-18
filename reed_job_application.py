from vision_bot import BrowserVisionBot

# Initialize bot with GIF recording enabled
bot = BrowserVisionBot(save_gif=True, model_name="gpt-5-nano")
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
        bot.act("if see: a sign in button/link/text is visible (look for a button like 'Sign in' or 'Login' in the page.) then: click: the sign in button (look for a button like 'Sign in' or 'Login' in the page.)")
        bot.act("if see: on sign in page then: ref: sign_in_without_password_commands")

bot.register_prompts([
        "click: an ios job listing button/link eg 'IOS Developer', 'Senior IOS Developer', 'Junior IOS Developer', etc (ignore generic links like 'ios jobs' or 'ios developer jobs' or 'ios developer jobs in london' or 'ios developer jobs in remote')",
], "click_job_command", max_retries=1, max_attempts=1)

bot.register_prompts([
        "dedup: enable",
        "ref: click_job_command",
        "dedup: disable",
        "click: the apply button (look for a button like 'Apply' or 'Apply Now' in the job listing. it could also look like an Easy Apply button, those are also valid)",
        "click: the button to submit the application (look for a button like 'Submit' or 'Submit Application' in modal)",
        "press: escape"
], "click_ios_job_action",
        additional_context="the job listing will be in a card view. also look for text surrounding a list of relevant job listings that are relevant to the search, e.g 'IOS Developer', 'Senior IOS Developer', 'Junior IOS Developer', etc",
        target_context_guard="only click a job listing if it its card does NOT have an 'Applied' status or an applied tag",
        confirm_before_interaction=True,
        all_must_be_true=True,
        command_id="click_ios_job_action")

bot.register_prompts([
        "if see: there is an ios job listing card without an 'Applied' status or an applied tag then: ref: click_ios_job_action else: scroll: down 600px"
], "click_ios_job_action_loop",
        additional_context="look for text surrounding a list of relevant job listings that are relevant to the search, e.g 'IOS Developer', 'Senior IOS Developer', 'Junior IOS Developer', etc. ignore generic links like 'ios jobs' or 'ios developer jobs' or 'ios developer jobs in london' or 'ios developer jobs in remote'. ignore jobs like 'Frontend Developer', 'Backend Developer', 'Full Stack Developer', 'Software Engineer', 'Senior Software Engineer', 'Junior Software Engineer', etc",
        command_id="click_ios_job_action_loop")

# If cookie banner is visible, click the button to accept cookies
bot.act("defer")
# bot.on_new_page_load(["if see: a cookie banner/dialog or privacy policy banner/dialog is visible then: click: the button to accept cookies (look for a button like 'Accept' or 'Accept all' in the cookie banner)"], command_id="cookie_banner_commands")
# sign_in_without_password_commands(bot)

# Apply to all available jobs, then click pagination
bot.act("while: not at bottom of the page do: ref: click_ios_job_action_loop", modifier=["page"])

# # Scroll to the top of the page
bot.act("scroll: to top")
bot.act("while: pagination control with previous and next buttons is not visible do: scroll: down 600px", additional_context="The pagination control must have a 'Previous' and 'Next' button", modifier=["see"])
bot.act("click: the next button in the pagination control", additional_context="The pagination control must have a 'Previous' and 'Next' button")
# End the bot session (performs cleanup and terminates)
bot.end()

input("Press enter to continue")