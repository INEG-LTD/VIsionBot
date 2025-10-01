from vision_bot import BrowserVisionBot

bot = BrowserVisionBot()
bot.start()
bot.goto("https://www.reed.co.uk/jobs/ios-developer-jobs")
bot.default_interpretation_mode = "semantic"

bot.register_prompts([
        "form: type 'eromoseleinegbe@gmail.com' into: the email input field (look for a input field like 'Email' or 'Email Address' in the page.",
        "click: the sign in without password button",
        "defer"
], "sign_in_without_password_commands", all_must_be_true=False)
bot.register_prompts([
    "click: the apply button: look for a button like 'Apply' or 'Apply Now' in the job listing. it could also look like an Easy Apply button, those are also valid",
    "click: the button to submit the application: look for a button like 'Submit' or 'Submit Application' in modal.",
    "press: escape"
], "apply_button_commands", all_must_be_true=False)
bot.register_prompts([
    "press: escape"
], "escape_button_commands", all_must_be_true=False)
bot.register_prompts([
        "dedup: enable",
        "click: an ios job listing button/link",
        "dedup: disable",
        "click: the apply button: look for a button like 'Apply' or 'Apply Now' in the job listing. it could also look like an Easy Apply button, those are also valid",
        "click: the button to submit the application: look for a button like 'Submit' or 'Submit Application' in modal.",
        "press: escape"
], "click_ios_job_action",
        additional_context="look for text surrounding a list of relevant job listings that are relevant to the search, e.g 'IOS Developer', 'Senior iOS Developer', 'Junior iOS Developer', etc",
        target_context_guard="only click if it its card does NOT have an 'Applied' status or an applied tag",
        confirm_before_interaction=True)

# If cookie banner is visible, click the button to accept cookies
bot.on_new_page_load(["if: cookie banner visible then: click: the button to accept cookies: look for a button like 'Accept' or 'Accept all' in the cookie banner"])

# If user is not logged in, click the button to login
bot.act("if: a sign in button/link/text is visible (look for a button like 'Sign in' or 'Login' in the page.) then: click: the sign in button (look for a button like 'Sign in' or 'Login' in the page.)")
bot.act("if: on sign in page then: ref: sign_in_without_password_commands")

bot.act("for 3 times: ref: click_ios_job_action")
input("Press enter to continue")