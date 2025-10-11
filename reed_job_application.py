from action_executor import PostActionContext
from ai_utils import answer_question_with_vision
from vision_bot import BrowserVisionBot
# from vision_bot_async import AsyncBrowserVisionBot

# Initialize bot with GIF recording enabled
bot = BrowserVisionBot()
bot.start()
bot.goto("https://www.reed.co.uk/")
bot.default_interpretation_mode = "semantic"

def post_action_callback(ctx: PostActionContext):
        if ctx.command_id == "click_ios_job_action_cmd6":
                print("About to click on the next job listing")
                
                page_screenshot = bot.page.screenshot(type="jpeg", quality=35, full_page=False)
                if answer_question_with_vision("is there a pagination control visible? so that we can go to the next page? it typically is numbered and has a next button", page_screenshot):
                        bot.queue_action(
                                "click: the next button (look for a button like 'Next' or 'Next Page' in the pagination control or the corresponding next number button)",
                                command_id="go_to_next_page_action",
                                priority=1,  # Higher priority
                                metadata={"triggered_by": ctx.command_id, "reason": "go_to_next_page_action"}
                        )
                        return
                
bot.action_executor.register_post_action_callback(post_action_callback)

bot.register_prompts([
        "form: type 'eromoseleinegbe@gmail.com' into: the email input field (look for a input field like 'Email' or 'Email Address' in the page.",
        "click: the sign in without password button",
        "defer"
], "sign_in_without_password_commands", all_must_be_true=False)
bot.register_prompts([
        "dedup: enable",
        "click: an ios job listing button/link",
        "dedup: disable",
        "click: the apply button (look for a button like 'Apply' or 'Apply Now' in the job listing. it could also look like an Easy Apply button, those are also valid)",
        "click: the button to submit the application (look for a button like 'Submit' or 'Submit Application' in modal)",
        "press: escape"
], "click_ios_job_action",
        additional_context="look for text surrounding a list of relevant job listings that are relevant to the search, e.g 'IOS Developer', 'Senior IOS Developer', 'Junior IOS Developer', etc",
        target_context_guard="only click a job listing if it its card does NOT have an 'Applied' status or an applied tag",
        confirm_before_interaction=True,
        command_id="click_ios_job_action")

# If cookie banner is visible, click the button to accept cookies
bot.on_new_page_load(["if: a cookie banner/dialog is visible then: click: the button to accept cookies (look for a button like 'Accept' or 'Accept all' in the cookie banner)"], command_id="cookie_banner_commands")

# Search for ios developer jobs in london or remote
bot.act("form: type 'ios developer' into the what field and 'london or remote' into the where field")
bot.act("click: the search jobs button")

# If user is not logged in, click the button to login and sign in without password
bot.act("if: a sign in button/link/text is visible (look for a button like 'Sign in' or 'Login' in the page.) then: click: the sign in button (look for a button like 'Sign in' or 'Login' in the page.)")
bot.act("if: on sign in page then: ref: sign_in_without_password_commands")

# Apply to the first 3 jobs that haven't been applied to yet
bot.act("for 3 times: ref: click_ios_job_action", command_id="apply-jobs-loop")

# End the bot session (performs cleanup and terminates)
gif_path = bot.end()

input("Press enter to continue")