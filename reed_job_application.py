from vision_bot import BrowserVisionBot

# Initialize bot with GIF recording enabled and high reasoning level
bot = BrowserVisionBot(save_gif=True, model_name="gpt-5-nano", reasoning_level="low")
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

def sign_up(bot: BrowserVisionBot):
        bot.act("click: the register cv button")
        
        # Request details from user
        email = input("Enter your email: ")
        password = input("Enter your password: ")
        
        bot.act(f"form: type '{email}' into the email input field")
        bot.act(f"form: type '{password}' into the password input field")
        bot.act("click: the continue button")
        
        bot.act("defer 3")
        
        bot.act("click: the button to choose a cv")
        
        first_name = "Chris"
        last_name = "Reed"
        contact_number = "07400000000"
        country = "United Kingdom"
        city = "London"
        current_job_title = "IOS Developer"
        eligible_to_work_in_uk = "Yes"
        desired_job_title = "IOS Developer"
        desired_minimum_salary = "£50,000"
        desired_work_location = "London"
        
        bot.act(f"form: type '{first_name}' into the first name input field")
        bot.act(f"form: type '{last_name}' into the last name input field")
        bot.act(f"form: type '{contact_number}' into the contact number input field")
        bot.act(f"form: select '{country}' from the country dropdown")
        bot.act(f"form: type '{city}' into the city input field")
        bot.act(f"form: type '{current_job_title}' into the current job title input field")
        bot.act(f"form: type '{eligible_to_work_in_uk}' into the eligible to work in uk input field")
        bot.act(f"form: type '{desired_job_title}' into the desired job title input field")
        bot.act(f"form: type '{desired_minimum_salary}' into the desired minimum salary input field")
        bot.act(f"form: select '{desired_work_location}' from the desired work location dropdown")
        bot.act("click: the create account button")
         
bot.register_prompts([
        "click: the button to submit the application (look for a button like 'Submit' or 'Submit Application' in modal)",
        "press: escape"
], "submit_application_command", max_retries=1, max_attempts=1, all_must_be_true=False)

bot.register_prompts([
        "click: the apply button (look for a button like 'Apply' or 'Apply Now' in the job listing. it could also look like an Easy Apply button, those are also valid)",
        "if see: the button to submit the application (look for a button like 'Submit' or 'Submit Application' in modal) then: ref: submit_application_command else: press: escape"
], "apply_to_job_command", max_retries=1, max_attempts=1, all_must_be_true=True)

bot.register_prompts([
        "if see: an apply button (look for a button like 'Apply' or 'Apply Now' in the job listing. it could also look like an Easy Apply button, those are also valid) then: ref: apply_to_job_command else: press: escape"
], "if_apply_to_job_command", max_retries=1, max_attempts=1)

bot.register_prompts([
        "click: an ios job listing button/link eg 'IOS Developer', 'Senior IOS Developer', 'Junior IOS Developer', etc (ignore generic links like 'ios jobs' or 'ios developer jobs' or 'ios developer jobs in london' or 'ios developer jobs in remote')",
], "click_job_command", 
        additional_context="if the element about to be clicked has a 'Applied' status or an applied tag, then ignore it. only job listings that have 'IOS' in the job title are valid, e.g 'IOS Developer', 'Senior IOS Developer', 'Junior IOS Developer', etc. ignore generic links like 'ios jobs' or 'ios developer jobs' or 'ios developer jobs in london' or 'ios developer jobs in remote'. ignore jobs like 'Frontend Developer', 'Backend Developer', 'Full Stack Developer', 'Software Engineer', 'Senior Software Engineer', 'Junior Software Engineer', etc",
        max_retries=1, max_attempts=1)

bot.register_prompts([
        "dedup: enable",
        "ref: click_job_command",
        "dedup: disable",
        "ref: if_apply_to_job_command",
], "click_ios_job_action",
        target_context_guard="only click a job listing if it its card does NOT have an 'Applied' status or an applied tag and has 'IOS' in the job title and is not plural (e.g 'IOS Developers' or 'IOS Developer Jobs in London' or 'IOS Developer Jobs in Remote')",
        confirm_before_interaction=True,
        all_must_be_true=True,
        command_id="click_ios_job_action")

bot.register_prompts([
        "if see: there is an ios job listing card without an 'Applied' label near it and has 'IOS' in the job title and is not plural (e.g 'IOS Developers' or 'IOS Developer Jobs in London' or 'IOS Developer Jobs in Remote') then: ref: click_ios_job_action else: scroll: down 600px"
], "click_ios_job_action_loop",
        additional_context="if the element about to be clicked has a 'Applied' status or an applied tag, then ignore it. only job listings that have 'IOS' in the job title are valid, e.g 'IOS Developer', 'Senior IOS Developer', 'Junior IOS Developer', etc. ignore generic links like 'ios jobs' or 'ios developer jobs' or 'ios developer jobs in london' or 'ios developer jobs in remote'. ignore jobs like 'Frontend Developer', 'Backend Developer', 'Full Stack Developer', 'Software Engineer', 'Senior Software Engineer', 'Junior Software Engineer', etc",
        command_id="click_ios_job_action_loop")

# If cookie banner is visible, click the button to accept cookies
bot.act("defer 3")
bot.on_new_page_load(["if see: there is a cookie banner/dialog or privacy policy banner/dialog with 'Accept' or 'Accept all' text is visible then: click: the button to accept cookies (look for a button like 'Accept' or 'Accept all' in the cookie banner)"], command_id="cookie_banner_commands")
sign_in_without_password_commands(bot)

# Apply to all available jobs, then click pagination
bot.act("while: not at bottom of the page do: ref: click_ios_job_action_loop continue_on_failure", modifier=["page"])

# # Scroll to the top of the page
bot.act("scroll: to top")
bot.act("while: pagination control with previous and next buttons is not visible do: scroll: down 600px", additional_context="The pagination control must have a 'Previous' and 'Next' button", modifier=["see"])
bot.act("click: the next button in the pagination control", additional_context="The pagination control must have a 'Previous' and 'Next' button")
# End the bot session (performs cleanup and terminates)
bot.end()

input("Press enter to continue")