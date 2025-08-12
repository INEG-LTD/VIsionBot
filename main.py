#!/usr/bin/env python3
"""
Job Application Automation Engine - Main Entry Point

This script provides a setup process for new users to configure their preferences
and then run the automation engine.
"""

import os
import sys
from user_setup import setup_process, load_config, show_config_summary

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
        print("\n📁 Configuration file found!")
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
        print("[1] View current configuration")
        print("[2] Edit configuration")
        print("[3] Start automation")
        print("[4] Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            show_config_summary(config)
            
        elif choice == "2":
            print("\n🔄 To edit configuration, you can:")
            print("   - Edit user_config.json directly")
            print("   - Delete user_config.json and restart for new setup")
            print("   - Run setup again")
            
        elif choice == "3":
            print("\n🚀 Starting automation...")
            print("Note: This will use your saved configuration.")
            
            # Here you would integrate with the existing automation engine
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
                    print("Starting job search...")
                    bot.find_jobs()
                else:
                    print("❌ Failed to start browser")
                    
            except ImportError as e:
                print(f"❌ Automation engine not available: {e}")
                print("Make sure all dependencies are installed.")
            except Exception as e:
                print(f"❌ Error starting automation: {e}")
                
        elif choice == "4":
            print("\n👋 Goodbye! Happy job hunting!")
            break
            
        else:
            print(f"❌ Invalid choice: {choice}")

if __name__ == '__main__':
    main()