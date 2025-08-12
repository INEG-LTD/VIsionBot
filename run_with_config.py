#!/usr/bin/env python3
"""
Run the Job Application Automation Engine with Configuration

This script demonstrates how to use the config.py file with the existing
automation engine. It loads your preferences and initializes the bot.
"""

import sys
import os
from config import get_all_preferences, validate_config, print_config_summary

def main():
    """Main function to run automation with config"""
    
    print("🚀 Job Application Automation Engine - Config Integration")
    print("=" * 70)
    
    # 1. Load and validate configuration
    print("\n1️⃣ Loading your configuration...")
    try:
        preferences = get_all_preferences()
        print(f"✅ Loaded {len(preferences)} preferences")
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return False
    
    # 2. Validate configuration
    print("\n2️⃣ Validating configuration...")
    errors = validate_config()
    if errors:
        print(f"⚠️ Found {len(errors)} configuration issues:")
        for error in errors:
            print(f"   - {error}")
        
        # Ask user if they want to continue
        response = input("\nDo you want to continue anyway? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("❌ Configuration validation failed. Please fix the issues and try again.")
            return False
        print("⚠️ Continuing with configuration issues...")
    else:
        print("✅ Configuration is valid!")
    
    # 3. Show configuration summary
    print("\n3️⃣ Your Configuration Summary:")
    print_config_summary()
    
    # 4. Initialize the automation engine
    print("\n4️⃣ Initializing automation engine...")
    try:
        # Import the automation engine
        from find_jobs import FindJobsBot
        
        # Initialize with your preferences
        bot = FindJobsBot(
            headless=preferences.get('headless', False),
            preferences=preferences
        )
        print("✅ Automation engine initialized successfully!")
        
        # 5. Show what's ready to run
        print("\n5️⃣ Ready to start automation!")
        print(f"   - Max jobs to find: {preferences.get('max_jobs_to_find', 5)}")
        print(f"   - Target locations: {', '.join(preferences.get('locations', [])[:3])}")
        print(f"   - Job titles: {', '.join(preferences.get('job_titles', [])[:3])}")
        print(f"   - Skills: {', '.join(preferences.get('required_skills', [])[:3])}")
        
        # 6. Ask user what they want to do
        print("\n6️⃣ What would you like to do?")
        print("   [1] Start job search (find_jobs)")
        print("   [2] Test configuration only")
        print("   [3] Show detailed preferences")
        print("   [4] Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == "1":
            print("\n🚀 Starting job search...")
            print("   Note: This will open a browser and start the automation process.")
            print("   Press Ctrl+C to stop at any time.")
            
            # Start the browser
            if bot.start_browser():
                print("✅ Browser started successfully!")
                
                # Start job search
                try:
                    bot.find_jobs()
                except KeyboardInterrupt:
                    print("\n⏹️ Job search stopped by user")
                except Exception as e:
                    print(f"❌ Error during job search: {e}")
                finally:
                    # Clean up
                    if bot.browser:
                        bot.browser.close()
                    if bot.playwright:
                        bot.playwright.stop()
            else:
                print("❌ Failed to start browser")
                
        elif choice == "2":
            print("\n✅ Configuration test completed successfully!")
            
        elif choice == "3":
            print("\n📋 Detailed Preferences:")
            print("=" * 50)
            for key, value in preferences.items():
                if isinstance(value, list) and len(value) > 5:
                    print(f"{key}: {value[:3]}... ({len(value)} total)")
                else:
                    print(f"{key}: {value}")
                    
        elif choice == "4":
            print("\n👋 Goodbye!")
            
        else:
            print(f"\n❌ Invalid choice: {choice}")
            
    except ImportError as e:
        print(f"❌ Failed to import automation engine: {e}")
        print("   Make sure all dependencies are installed:")
        print("   pip install -r requirements.txt")
        return False
        
    except Exception as e:
        print(f"❌ Error initializing automation engine: {e}")
        return False
    
    return True

def show_usage():
    """Show usage information"""
    print("\n📖 Usage Information:")
    print("=" * 50)
    print("This script integrates your config.py preferences with the automation engine.")
    print("\nTo customize your preferences:")
    print("1. Edit config.py with your personal information")
    print("2. Update job titles, locations, and skills")
    print("3. Set your salary expectations and preferences")
    print("4. Configure browser and AI settings")
    print("\nFor detailed help, see CONFIG_README.md")

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            show_usage()
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⏹️ Script interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        show_usage()
        sys.exit(1)
