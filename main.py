#!/usr/bin/env python3
"""
Job Application Automation Engine - Main Entry Point

This script provides a setup process for new users to configure their preferences
and then run the automation engine.
"""

import os
import sys
from user_setup import setup_process, load_config, show_config_summary
from rich import print as rprint

def print_banner():
    """Print the application banner"""
    try:
        from pyfiglet import Figlet
        f = Figlet(font='slant')
        print(f.renderText('Job Applyer'))
    except ImportError:
        print("🚀 Job Application Automation Engine")
        print("=" * 50)

def main():
    """Main function"""
    print_banner()
    
    # Check if this is first run
    if not os.path.exists("user_config.json"):
        print("\n🆕 First time setup detected!")
        config = setup_process()
        if not config:
            print("❌ Setup failed. Exiting.")
            sys.exit(1)
    else:
        print("\n> 📁 Configuration file found!")
        config = load_config()
        if not config:
            print("❌ Failed to load configuration. Starting setup...")
            config = setup_process()
            if not config:
                print("❌ Setup failed. Exiting.")
                sys.exit(1)
    
    # Show main menu
    while True:
        print("\n" + "=" * 50)
        print("🎯 Job Application Automation Engine")
        print("=" * 50)
        print("What would you like to do?")
        rprint("[1] [bold][dim]Full Job Application [/dim][red](in development)[/red][dim] | Job board → Filter Acceptable Jobs → Apply to acceptable jobs[/dim]")
        rprint("    [italic]⮑ Complete end-to-end automation from job discovery to application[/italic]")
        rprint("[2] [bold]Apply to 'Easy apply' jobs[/bold]")
        rprint("    [italic]⮑ Automatically apply to jobs that have an 'Easy Apply' button or similar[/italic]")
        rprint("[3] [bold]Apply directly to job (With job form)[/bold]")
        rprint("    [italic]⮑ Paste a job link and the system will automatically fill the form[/italic]")
        rprint("[4] [bold]Edit/Delete config[/bold]")
        rprint("[5] [bold]Exit[/bold]")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            print("\n🚀 Full Job Application Pipeline")
            print("=" * 40)
            print("This feature is currently in development.")
            print("It will include:")
            print("   - Job board scraping and monitoring")
            print("   - AI-powered job filtering and scoring")
            print("   - Automated application submission")
            print("   - Progress tracking and analytics")
            print("\n⚠️ Coming soon! Stay tuned for updates.")
            
        elif choice == "2":
            print("\n⚡ Easy Apply")
            print("=" * 20)
            print("Quick application to jobs you've already found.")
            print("This will use your saved configuration to:")
            print("   - Fill out application forms automatically")
            print("   - Upload your resume and cover letter")
            print("   - Submit applications with one click")
            print("\n🚀 Starting Easy Apply mode...")
            
            try:
                from config import get_all_preferences
                from find_jobs import FindJobsBot
                
                # Convert our config to the format expected by the automation engine
                preferences = get_all_preferences()
                preferences.update(config)  # Override with user config
                
                bot = FindJobsBot(
                    headless=preferences.get('headless', False),
                    preferences=preferences
                )
                
                if bot.start_browser():
                    print("✅ Browser started successfully!")
                    print("Ready for Easy Apply - navigate to a job application form")
                    print("The system will automatically detect and fill forms")
                else:
                    print("❌ Failed to start browser")
                    
            except ImportError as e:
                print(f"❌ Automation engine not available: {e}")
                print("Make sure all dependencies are installed.")
            except Exception as e:
                print(f"❌ Error starting automation: {e}")
                
        elif choice == "3":
            print("\n📝 Job Form Filler")
            print("=" * 20)
            print("Fill out job application forms with your saved preferences.")
            print("This will:")
            print("   - Auto-fill personal information")
            print("   - Populate job-specific fields")
            print("   - Handle various form layouts")
            print("\n🚀 Starting Job Form Filler...")
            
            try:
                from config import get_all_preferences
                from fill_and_submit_job_form import ApplicationFiller
                
                # Convert our config to the format expected by the automation engine
                preferences = get_all_preferences()
                preferences.update(config)  # Override with user config
                
                print("✅ Job Form Filler initialized!")
                print("Navigate to a job application form and the system will")
                print("automatically detect and fill form fields.")
                
            except ImportError as e:
                print(f"❌ Form filler not available: {e}")
                print("Make sure all dependencies are installed.")
            except Exception as e:
                print(f"❌ Error initializing form filler: {e}")
                
        elif choice == "4":
            print("\n⚙️ Configuration Management")
            print("=" * 30)
            print("What would you like to do with your configuration?")
            print("[a] View current configuration")
            print("[b] Edit configuration manually")
            print("[c] Delete configuration and restart setup")
            print("[d] Back to main menu")
            
            config_choice = input("\nEnter your choice (a-d): ").strip().lower()
            
            if config_choice == "a":
                show_config_summary(config)
                
            elif config_choice == "b":
                print("\n🔄 To edit configuration manually:")
                print("   1. Edit user_config.json in a text editor")
                print("   2. Save the file")
                print("   3. Restart the application to load changes")
                print("   4. Or use option [c] to restart setup")
                
            elif config_choice == "c":
                print("\n🗑️ Delete current configuration?")
                confirm = input("This will remove all your preferences. Continue? (y/N): ").strip().lower()
                if confirm in ['y', 'yes']:
                    try:
                        os.remove("user_config.json")
                        print("✅ Configuration deleted successfully!")
                        print("🔄 Please restart the application to run setup again.")
                        break
                    except Exception as e:
                        print(f"❌ Error deleting configuration: {e}")
                else:
                    print("✅ Configuration deletion cancelled.")
                    
            elif config_choice == "d":
                print("↩️ Returning to main menu...")
                
            else:
                print(f"❌ Invalid choice: {config_choice}")
                
        elif choice == "5":
            print("\n👋 Goodbye! Happy job hunting!")
            break
            
        else:
            print(f"❌ Invalid choice: {choice}")
            print("Please enter a number between 1 and 5.")

if __name__ == '__main__':
    main()