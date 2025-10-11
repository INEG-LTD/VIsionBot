"""
Reed Job Application - With Advanced Command Tracking

Features:
- Saves session ledger for later analysis
- Compares with previous runs
- Integrates with logger for unified tracking
"""
from vision_bot import BrowserVisionBot
from command_ledger import CommandLedger, CommandStatus
from datetime import datetime
from pathlib import Path

# Initialize bot with GIF recording enabled
bot = BrowserVisionBot(save_gif=True)
bot.start()

# Enable logger integration for unified tracking
bot.command_ledger.enable_logger_integration(bot.logger)
print("✅ Command tracking enabled - all commands will be logged\n")

# Set default interpretation mode
bot.goto("https://www.reed.co.uk/")
bot.default_interpretation_mode = "semantic"

# Register command sequences
bot.register_prompts([
    "form: type 'eromoseleinegbe@gmail.com' into: the email input field (look for a input field like 'Email' or 'Email Address' in the page.",
    "click: the sign in without password button",
    "defer"
], "sign_in_without_password_commands", all_must_be_true=False, command_id="signin-flow")

bot.register_prompts([
    "click: the apply button (look for a button like 'Apply' or 'Apply Now' in the job listing. it could also look like an Easy Apply button, those are also valid)",
    "click: the button to submit the application: look for a button like 'Submit' or 'Submit Application' in modal.",
    "press: escape"
], "apply_button_commands", all_must_be_true=False, command_id="apply-flow")

bot.register_prompts([
    "press: escape"
], "escape_button_commands", all_must_be_true=False, command_id="escape-flow")

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
    confirm_before_interaction=True,
    command_id="ios-job-action")

# If cookie banner is visible, click the button to accept cookies
bot.on_new_page_load(["if: a cookie banner/dialog is visible then: click: the button to accept cookies: look for a button like 'Accept' or 'Accept all' in the cookie banner"])

# Search for ios developer jobs in london or remote
bot.act("form: type 'ios developer' into the what field and 'london or remote' into the where field", command_id="search-form")
bot.act("click: the search jobs button", command_id="search-btn")

# If user is not logged in, click the button to login and sign in without password
bot.act("if: a sign in button/link/text is visible (look for a button like 'Sign in' or 'Login' in the page.) then: click: the sign in button (look for a button like 'Sign in' or 'Login' in the page.)", command_id="check-login")
bot.act("if: on sign in page then: ref: sign_in_without_password_commands", command_id="signin-if")

# Apply to the first 3 jobs that haven't been applied to yet
bot.act("for 3 times: ref: click_ios_job_action", command_id="apply-loop")

# End the bot session
gif_path = bot.end()

print("\n" + "="*80)
print("SESSION COMPLETED")
print("="*80)

# Save the ledger for later analysis
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
ledger_dir = Path("ledgers")
ledger_dir.mkdir(exist_ok=True)

session_file = ledger_dir / f"reed_session_{timestamp}.json"
summary_file = ledger_dir / f"reed_session_{timestamp}_summary.txt"

bot.command_ledger.save_to_file(str(session_file))
bot.command_ledger.export_summary(str(summary_file))

print(f"\n💾 Session saved to: {session_file}")
print(f"📄 Summary saved to: {summary_file}")

# Print session statistics
stats = bot.command_ledger.get_stats()
print(f"\n📊 Session Statistics:")
print(f"   Total commands: {stats['total_commands']}")
print(f"   Total duration: {stats['total_duration']:.2f}s")
print(f"   Average duration: {stats['average_duration']:.3f}s")
print(f"   Status breakdown: {stats['by_status']}")

# Check for failed commands
failed = bot.command_ledger.filter_records(status=CommandStatus.FAILED)
if failed:
    print(f"\n⚠️ Failed Commands ({len(failed)}):")
    for cmd in failed:
        print(f"   ❌ {cmd.command}")
        if cmd.error_message:
            print(f"      Error: {cmd.error_message}")
else:
    print("\n✅ All commands completed successfully!")

# Compare with previous run if it exists
previous_files = sorted(ledger_dir.glob("reed_session_*.json"))
if len(previous_files) >= 2:
    print(f"\n🔍 Comparing with previous run...")
    try:
        comparison = CommandLedger.load_and_compare(
            str(previous_files[-2]),  # Previous run
            str(previous_files[-1])   # Current run
        )
        comparison.print_summary()
    except Exception as e:
        print(f"   Could not compare: {e}")
else:
    print(f"\n💡 This is your first tracked session. Run again to see comparisons!")

print("\n" + "="*80)
print("Session log and GIF recording saved")
print("="*80)

input("\nPress enter to exit...")

