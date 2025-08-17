#!/usr/bin/env python3
"""
Job Application Automation Engine - Main Entry Point

This script provides a setup process for new users to configure their preferences
and then run the automation engine.
"""

import os
import sys
from bot_utils import start_browser, debug_mode
from fill_and_submit_job_form import ApplicationFiller
from find_jobs import FindJobsBot
from user_setup import setup_process, load_config, show_config_summary
from InquirerPy import inquirer
from InquirerPy.validator import PathValidator
from InquirerPy.base.control import Choice
from rich import print as rprint
from rich.progress import Progress
from rich.live import Live
from terminal import term,banner, clear_screen, clear_screen_preserve_banner, banner_lines
from user_setup import show_config_summary
import requests
import time
from yaspin import yaspin

def check_setup() -> bool:
    """Check if the setup process has been run"""
    try:
        if not os.path.exists("user_config.json"):
            return False
        return True
    except Exception as e:
        print(f"❌ Error checking setup: {e}")
        return False

def run_setup_check():
    if not check_setup():
        print(term.move_y((term.height // 2)))

        rprint(term.center("[bold] First time setup detected"))
        rprint(term.center("[dim]Please follow the prompts to configure your preferences - this will take a few minutes, and you can always edit them later.\n\n"))
        print(term.black_on_darkkhaki(term.center("press any key to continue...")))
        
        with term.cbreak(), term.hidden_cursor():
            term.inkey()
            
        config = setup_process()
        if not config:
            print("❌ Setup failed. Exiting.")
            sys.exit(1)
    else:
        config = load_config()
        if not config:
            print("❌ Failed to load configuration. Starting setup...")
            config = setup_process()
            if not config:
                print("❌ Setup failed. Exiting.")
                sys.exit(1)

def wait_and_return_to_menu(menu_function):
    print("Press any key to continue...")
    with term.cbreak(), term.hidden_cursor():
        term.inkey()
        
    clear_screen()
    print(banner)
    
    menu_function()
    
def edit_delete_option():
    rprint("[bold]Edit/Delete config - Manage your configuration settings\n")
    edit_delete_choices = [
        Choice(value="a", name="View current configuration"),
        Choice(value="b", name="Edit configuration manually"),
        Choice(value="c", name="Delete configuration and restart setup"),
        Choice(value="d", name="Back to main menu")
    ]
    
    choice = inquirer.select(
        message="What would you like to do with your configuration?",
        choices=edit_delete_choices,
        default="a"
    ).execute()
    
    if choice == "a":
        config = load_config()
        if not config:
            print("❌ Failed to load configuration. Starting setup...")
            main()
            return
        show_config_summary(config)
            
        wait_and_return_to_menu(edit_delete_option)
        return
    
    elif choice == "b":
        rprint("\n[bold]To edit configuration manually:[/bold]")
        print("   1. Edit user_config.json in a text editor")
        print("   2. Save the file")
        print("   3. Restart the application to load changes\n")
        
        wait_and_return_to_menu(edit_delete_option)
        return
    elif choice == "c":
        confirm = inquirer.confirm(
            message="Are you sure you want to delete your configuration? Continue?",
            long_instruction="This will remove all your preferences and you will need to restart the setup process.",
            default=False
        ).execute()
        if confirm:
            os.remove("user_config.json")
            print("✅ Configuration deleted successfully!")
            print("🔄 Please restart the application to run setup again.")
            return
        else:
            rprint("[green]Configuration deletion cancelled.[/green]\n")
            
        wait_and_return_to_menu(edit_delete_option)
        return
    
    elif choice == "d":
        print("↩️ Returning to main menu...")
        main()
        return

def verify_url(url: str) -> bool:
    """Verify if the URL is valid"""
    with yaspin(text="Verifying URL...", color="cyan") as spinner:
        try:
            # add https:// to the url if it's not there
            if not url.startswith("https://"):
                url = "https://" + url
                
                
            response = requests.get(url, timeout=10)
            
            spinner.hide()
            
            return response.status_code == 200
        except Exception as e:
            spinner.hide()
            if debug_mode:
                print(f"❌ Error verifying URL: {e}")
            return False

def main():
    """Main function"""
    clear_screen()
    print(banner)
    
    # Check if this is first run
    run_setup_check()
    
    rprint("[bold]Main Menu\n")
    # Main menu choices
    choices = [
        Choice(value="1", name="Full Job Application (in development) - Complete end-to-end automation from job discovery to application"),
        Choice(value="2", name="Apply to 'Easy apply' jobs - Automatically apply to jobs with Easy Apply buttons"),
        Choice(value="3", name="Apply directly to job (With job form) - Paste a job link and auto-fill the form"),
        Choice(value="4", name="Edit/Delete config - Manage your configuration settings"),
        Choice(value="5", name="Exit - Close the application")
    ]
    
    choice = inquirer.select(
        message="What would you like to do?",
        choices=choices,
        default="2"
    ).execute()
    
    # Clear screen while preserving banner
    clear_screen_preserve_banner(banner_lines)
    
    if choice == "1":
        print("Full Job Application (in development) - Complete end-to-end automation from job discovery to application")
    elif choice == "2":
        print("Apply to 'Easy apply' jobs - Automatically apply to jobs with Easy Apply buttons")
    elif choice == "3":
        print("Apply directly to job (With job form) - Paste a job link and auto-fill the form")
        
        test_url = "https://www.sumup.com/careers/positions/london-england-united-kingdom/ios/senior-ios-engineer-global-bank/8048304002/?gh_jid=8048304002&gh_src=jn5gvww32us"
        url = inquirer.text(
            message="Please enter the URL of the job you want to apply to",
            default=test_url if debug_mode else ""
        ).execute()
        
        if url:
            if not verify_url(url):
                rprint("[red]Invalid URL[/red]")
                wait_and_return_to_menu(main)
                return
            
            # add https:// to the url if it's not there
            if not url.startswith("https://"):
                url = "https://" + url
                
            config = load_config()
            
            if not config:
                clear_screen_preserve_banner(banner_lines)
                run_setup_check()
                return
            
            clear_screen_preserve_banner(banner_lines)
            
            browser, context, page = start_browser()
            page.goto(url, wait_until='domcontentloaded', timeout=15000)
            bot = ApplicationFiller(page=page, preferences=config)
            bot.run_main_algorithm()
            
            # bot.page = page
            
            # bot.page.goto(url, wait_until='domcontentloaded', timeout=15000)
            # bot.page.pause()
            
    elif choice == "4":
        edit_delete_option()
            
    elif choice == "5":
        print("👋 Goodbye! Happy job hunting!\n")
        return

    # with term.fullscreen():
    #     print("\n" + "=" * 50)
    #     print("🎯 Job Application Automation Engine")
    #     print("=" * 50)
        
    #     # Main menu choices
    #     choices = [
    #         Choice(value="1", name="Full Job Application (in development) - Complete end-to-end automation from job discovery to application"),
    #         Choice(value="2", name="Apply to 'Easy apply' jobs - Automatically apply to jobs with Easy Apply buttons"),
    #         Choice(value="3", name="Apply directly to job (With job form) - Paste a job link and auto-fill the form"),
    #         Choice(value="4", name="Edit/Delete config - Manage your configuration settings"),
    #         Choice(value="5", name="Exit - Close the application")
    #     ]
        
    #     choice = inquirer.select(
    #         message="What would you like to do?",
    #         choices=choices,
    #         default="2"
    #     ).execute()
        
    #     if choice == "1":
    #         print("\n🚀 Full Job Application Pipeline")
    #         print("=" * 40)
    #         print("This feature is currently in development.")
    #         print("It will include:")
    #         print("   - Job board scraping and monitoring")
    #         print("   - AI-powered job filtering and scoring")
    #         print("   - Automated application submission")
    #         print("   - Progress tracking and analytics")
    #         print("\n⚠️ Coming soon! Stay tuned for updates.")
            
    #     elif choice == "2":
    #         print("\n⚡ Easy Apply")
    #         print("=" * 20)
    #         print("Quick application to jobs you've already found.")
    #         print("This will use your saved configuration to:")
    #         print("   - Fill out application forms automatically")
    #         print("   - Upload your resume and cover letter")
    #         print("   - Submit applications with one click")
    #         print("\n🚀 Starting Easy Apply mode...")
            
    #         try:
    #             from config import get_all_preferences
    #             from find_jobs import FindJobsBot
                
    #             # Convert our config to the format expected by the automation engine
    #             preferences = get_all_preferences()
    #             preferences.update(config)  # Override with user config
                
    #             bot = FindJobsBot(
    #                 headless=preferences.get('headless', False),
    #                 preferences=preferences
    #             )
                
    #             if bot.start_browser():
    #                 print("✅ Browser started successfully!")
    #                 print("Ready for Easy Apply - navigate to a job application form")
    #                 print("The system will automatically detect and fill forms")
    #             else:
    #                 print("❌ Failed to start browser")
                    
    #         except ImportError as e:
    #             print(f"❌ Automation engine not available: {e}")
    #             print("Make sure all dependencies are installed.")
    #         except Exception as e:
    #             print(f"❌ Error starting automation: {e}")
                
    #     elif choice == "3":
    #         print("\n📝 Job Form Filler")
    #         print("=" * 20)
    #         print("Fill out job application forms with your saved preferences.")
    #         print("This will:")
    #         print("   - Auto-fill personal information")
    #         print("   - Populate job-specific fields")
    #         print("   - Handle various form layouts")
    #         print("\n🚀 Starting Job Form Filler...")
            
    #         try:
    #             from config import get_all_preferences
    #             from fill_and_submit_job_form import ApplicationFiller
                
    #             # Convert our config to the format expected by the automation engine
    #             preferences = get_all_preferences()
    #             preferences.update(config)  # Override with user config
                
    #             print("✅ Job Form Filler initialized!")
    #             print("Navigate to a job application form and the system will")
    #             print("automatically detect and fill form fields.")
                
    #         except ImportError as e:
    #             print(f"❌ Form filler not available: {e}")
    #             print("Make sure all dependencies are installed.")
    #         except Exception as e:
    #             print(f"❌ Error initializing form filler: {e}")
                
    #     elif choice == "4":
    #         print("\n⚙️ Configuration Management")
    #         print("=" * 30)
            
    #         # Configuration submenu
    #         config_choices = [
    #             Choice(value="a", name="View current configuration"),
    #             Choice(value="b", name="Edit configuration manually"),
    #             Choice(value="c", name="Delete configuration and restart setup"),
    #             Choice(value="d", name="Back to main menu")
    #         ]
            
    #         config_choice = inquirer.select(
    #             message="What would you like to do with your configuration?",
    #             choices=config_choices,
    #             default="a"
    #         ).execute()
            
    #         if config_choice == "a":
    #             show_config_summary(config)
                
    #         elif config_choice == "b":
    #             print("\n🔄 To edit configuration manually:")
    #             print("   1. Edit user_config.json in a text editor")
    #             print("   2. Save the file")
    #             print("   3. Restart the application to load changes")
    #             print("   4. Or use option [c] to restart setup")
                
    #         elif config_choice == "c":
    #             print("\n🗑️ Delete current configuration?")
    #             confirm = inquirer.confirm(
    #                 message="This will remove all your preferences. Continue?",
    #                 default=False
    #             ).execute()
                
    #             if confirm:
    #                 try:
    #                     os.remove("user_config.json")
    #                     print("✅ Configuration deleted successfully!")
    #                     print("🔄 Please restart the application to run setup again.")
    #                     return
    #                 except Exception as e:
    #                     print(f"❌ Error deleting configuration: {e}")
    #             else:
    #                 print("✅ Configuration deletion cancelled.")
                    
    #         elif config_choice == "d":
    #             print("↩️ Returning to main menu...")
                
    #     elif choice == "5":
    #         print("\n👋 Goodbye! Happy job hunting!")
    #         return

if __name__ == '__main__':
    main()