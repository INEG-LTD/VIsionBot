#!/usr/bin/env python3
"""
Example usage of the configuration file with the Job Application Automation Engine

This script demonstrates how to use the config.py file to customize
the automation engine's behavior.
"""

from config import get_all_preferences, validate_config, print_config_summary
from find_jobs import FindJobsBot

def main():
    """Main function demonstrating config usage"""
    
    print("🚀 Job Application Automation Engine - Configuration Example")
    print("=" * 70)
    
    # 1. Load and validate configuration
    print("\n1️⃣ Loading configuration...")
    preferences = get_all_preferences()
    
    # Validate the configuration
    errors = validate_config()
    if errors:
        print("❌ Configuration errors found:")
        for error in errors:
            print(f"   - {error}")
        print("\nPlease fix these errors in config.py before proceeding.")
        return
    
    print("✅ Configuration loaded successfully!")
    
    # 2. Print configuration summary
    print("\n2️⃣ Configuration Summary:")
    print_config_summary()
    
    # 3. Example: Initialize the bot with custom preferences
    print("\n3️⃣ Initializing automation bot...")
    
    # You can override specific preferences here if needed
    custom_preferences = preferences.copy()
    custom_preferences['max_jobs_to_find'] = 3  # Override for testing
    
    try:
        # Initialize the bot with your preferences
        bot = FindJobsBot(
            headless=preferences.get('headless', False),
            preferences=custom_preferences
        )
        print("✅ Bot initialized successfully!")
        
        # 4. Example: Access specific preferences
        print("\n4️⃣ Accessing specific preferences:")
        print(f"   - Job Titles: {', '.join(preferences['job_titles'][:3])}")
        print(f"   - Target Salary: {preferences['currency']} {preferences['salary_min']:,}+")
        print(f"   - Preferred Locations: {', '.join(preferences['locations'][:3])}")
        print(f"   - Key Skills: {', '.join(preferences['required_skills'][:3])}")
        
        # 5. Example: Modify preferences programmatically
        print("\n5️⃣ Modifying preferences programmatically:")
        
        # Add a new job title
        preferences['job_titles'].append('Machine Learning Engineer')
        print(f"   - Added new job title: Machine Learning Engineer")
        
        # Update salary expectations
        preferences['salary_min'] = 90000
        print(f"   - Updated minimum salary to: {preferences['currency']} {preferences['salary_min']:,}")
        
        # Add new skills
        preferences['required_skills'].extend(['TensorFlow', 'PyTorch'])
        print(f"   - Added new skills: TensorFlow, PyTorch")
        
        print("\n✅ Preferences updated successfully!")
        
        # 6. Example: Start the automation process
        print("\n6️⃣ Ready to start automation!")
        print("   To start the job search process, uncomment the following lines:")
        print("   bot.start_browser()")
        print("   bot.find_jobs()")
        
    except Exception as e:
        print(f"❌ Error initializing bot: {e}")
        print("   Make sure all dependencies are installed and config is valid.")

def show_config_structure():
    """Show the structure of the configuration"""
    print("\n📋 Configuration Structure:")
    print("=" * 50)
    
    sections = [
        "PERSONAL_INFO - Basic personal details",
        "ADDRESS_INFO - Address information", 
        "ADDITIONAL_INFO - Additional personal details",
        "DOCUMENT_PATHS - Resume, cover letter, photo paths",
        "JOB_TITLES - Job titles to search for",
        "LOCATIONS - Preferred locations",
        "SALARY_PREFERENCES - Salary expectations",
        "EMPLOYMENT_TYPES - Types of employment",
        "REQUIRED_SKILLS - Skills and technologies",
        "EXPERIENCE_LEVELS - Experience requirements",
        "REMOTE_FLEXIBILITY - Remote work preferences",
        "DESIRED_BENEFITS - Benefits you want",
        "EXCLUDE_KEYWORDS - Keywords to avoid",
        "VISA_PREFERENCES - Work authorization",
        "APPLICATION_SETTINGS - Automation behavior",
        "BROWSER_SETTINGS - Browser configuration",
        "AI_SETTINGS - AI model configuration",
        "LOGGING_SETTINGS - Logging and debugging",
        "WEBSITE_SETTINGS - Job site specific settings",
        "NOTIFICATION_SETTINGS - Email and notifications"
    ]
    
    for i, section in enumerate(sections, 1):
        print(f"{i:2d}. {section}")

if __name__ == "__main__":
    main()
    show_config_structure()
    
    print("\n" + "=" * 70)
    print("💡 Tips for customization:")
    print("   - Edit config.py to change your preferences")
    print("   - Run 'python config.py' to see your current config")
    print("   - Use 'python example_usage.py' to test your configuration")
    print("   - All changes in config.py will be automatically loaded")
    print("=" * 70)
