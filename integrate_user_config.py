#!/usr/bin/env python3
"""
Integration script showing how user configuration works with the existing config system

This script demonstrates how the user_config.json file integrates with the config.py
file to provide a seamless configuration experience.
"""

import json
import os
from config import get_all_preferences, validate_config, print_config_summary

def load_user_config():
    """Load user configuration from JSON file"""
    try:
        if os.path.exists("user_config.json"):
            with open("user_config.json", 'r') as f:
                return json.load(f)
        return None
    except Exception as e:
        print(f"❌ Error loading user config: {e}")
        return None

def merge_configs():
    """Merge default config with user config"""
    print("🔧 Configuration Integration Demo")
    print("=" * 50)
    
    # 1. Load default configuration
    print("\n1️⃣ Loading default configuration from config.py...")
    try:
        default_config = get_all_preferences()
        print(f"✅ Loaded {len(default_config)} default preferences")
    except Exception as e:
        print(f"❌ Failed to load default config: {e}")
        return None
    
    # 2. Load user configuration
    print("\n2️⃣ Loading user configuration from user_config.json...")
    user_config = load_user_config()
    if user_config:
        print(f"✅ Loaded {len(user_config)} user preferences")
    else:
        print("⚠️ No user configuration found")
        return default_config
    
    # 3. Merge configurations
    print("\n3️⃣ Merging configurations...")
    merged_config = default_config.copy()
    merged_config.update(user_config)  # User config overrides defaults
    print(f"✅ Merged configuration has {len(merged_config)} total preferences")
    
    # 4. Show what was overridden
    print("\n4️⃣ User overrides:")
    for key, value in user_config.items():
        if key in default_config:
            default_value = default_config[key]
            if default_value != value:
                print(f"   - {key}: {default_value} → {value}")
        else:
            print(f"   - {key}: (new) {value}")
    
    return merged_config

def show_integration_example():
    """Show how the integration works in practice"""
    print("\n🔗 Integration Example")
    print("=" * 30)
    
    # Load merged configuration
    config = merge_configs()
    if not config:
        print("❌ Failed to load configuration")
        return
    
    # Show final configuration
    print("\n📋 Final Merged Configuration:")
    # Note: print_config_summary() doesn't take parameters, it uses global config
    # So we'll show the merged config directly
    print(f"   - Name: {config.get('first_name', 'N/A')} {config.get('last_name', 'N/A')}")
    print(f"   - Email: {config.get('email', 'N/A')}")
    print(f"   - Phone: {config.get('phone', 'N/A')}")
    print(f"   - Job Titles: {', '.join(config.get('job_titles', [])[:3])}")
    print(f"   - Skills: {', '.join(config.get('required_skills', [])[:3])}")
    print(f"   - Max Jobs: {config.get('max_jobs_to_find', 'N/A')}")
    
    # Demonstrate specific overrides
    print("\n🎯 Key User Preferences:")
    print(f"   - Name: {config.get('first_name', 'N/A')} {config.get('last_name', 'N/A')}")
    print(f"   - Job Titles: {', '.join(config.get('job_titles', []))}")
    print(f"   - Skills: {', '.join(config.get('required_skills', []))}")
    print(f"   - Max Jobs: {config.get('max_jobs_to_find', 'N/A')}")
    
    # Validate merged configuration
    print("\n🔍 Validating merged configuration...")
    # Simple validation for merged config
    errors = []
    
    # Check required fields
    required_fields = ['first_name', 'last_name', 'email', 'phone']
    for field in required_fields:
        if not config.get(field):
            errors.append(f"Missing required field: {field}")
    
    # Check salary range
    if config.get('salary_min') and config.get('salary_max'):
        if config['salary_min'] > config['salary_max']:
            errors.append("Minimum salary cannot be greater than maximum salary")
    
    if errors:
        print(f"⚠️ Found {len(errors)} validation errors:")
        for error in errors:
            print(f"   - {error}")
    else:
        print("✅ Merged configuration is valid!")

def demonstrate_automation_integration():
    """Demonstrate how to use the merged config with automation"""
    print("\n🤖 Automation Integration Demo")
    print("=" * 40)
    
    config = merge_configs()
    if not config:
        return
    
    print("\n🚀 Ready to initialize automation engine with merged config!")
    print("   This would use your personal preferences while keeping all the")
    print("   default settings for features you haven't customized.")
    
    # Show what would be passed to the automation engine
    print("\n📤 Configuration passed to automation engine:")
    automation_keys = [
        'first_name', 'last_name', 'email', 'phone',
        'job_titles', 'locations', 'salary_min', 'required_skills',
        'max_jobs_to_find', 'headless', 'auto_apply'
    ]
    
    for key in automation_keys:
        if key in config:
            value = config[key]
            if isinstance(value, list) and len(value) > 3:
                print(f"   - {key}: {value[:3]}... ({len(value)} total)")
            else:
                print(f"   - {key}: {value}")

def main():
    """Main function"""
    print("🚀 User Configuration Integration Demo")
    print("=" * 60)
    
    # Check if user config exists
    if not os.path.exists("user_config.json"):
        print("❌ No user configuration found!")
        print("   Please run main.py first to create your configuration.")
        return
    
    # Show integration
    show_integration_example()
    
    # Show automation integration
    demonstrate_automation_integration()
    
    print("\n" + "=" * 60)
    print("💡 How it works:")
    print("   1. config.py provides sensible defaults for all features")
    print("   2. user_config.json contains your personal preferences")
    print("   3. The system merges them, with your preferences taking priority")
    print("   4. The automation engine uses the merged configuration")
    print("=" * 60)

if __name__ == "__main__":
    main()
