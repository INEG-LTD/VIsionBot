#!/usr/bin/env python3
"""
User Setup Module for Job Application Automation Engine

This module contains all the functions needed for the interactive setup process
that collects user preferences and saves them to a configuration file.
"""

import json
import os
import sys
from pathlib import Path

def get_user_input(prompt, default="", required=True, input_type=str):
    """Get user input with validation"""
    while True:
        if default:
            user_input = input(f"{prompt} [{default}]: ").strip()
            if not user_input:
                user_input = default
        else:
            user_input = input(f"{prompt}: ").strip()
        
        if not user_input and required:
            print("❌ This field is required. Please enter a value.")
            continue
        
        if not user_input and not required:
            return ""
        
        try:
            if input_type == int:
                return int(user_input)
            elif input_type == float:
                return float(user_input)
            elif input_type == bool:
                return user_input.lower() in ['y', 'yes', 'true', '1']
            else:
                return user_input
        except ValueError:
            print(f"❌ Invalid input. Please enter a valid {input_type.__name__}.")
            continue

def get_list_input(prompt, default=None):
    """Get a list of items from user"""
    print(f"\n{prompt}")
    print("Enter items one per line. Press Enter twice when done.")
    
    if default:
        print(f"Default: {', '.join(default)}")
        use_default = input("Use default? (y/N): ").strip().lower() in ['y', 'yes']
        if use_default:
            return default
    
    items = []
    while True:
        item = input(f"Item {len(items) + 1}: ").strip()
        if not item:
            if len(items) == 0:
                print("❌ At least one item is required.")
                continue
            break
        items.append(item)
    
    return items

def collect_personal_info():
    """Collect basic personal information"""
    print("\n👤 Personal Information")
    print("=" * 30)
    
    personal_info = {}
    
    personal_info['first_name'] = get_user_input("First Name", required=True)
    personal_info['last_name'] = get_user_input("Last Name", required=True)
    personal_info['email'] = get_user_input("Email Address", required=True)
    personal_info['phone'] = get_user_input("Phone Number", required=True)
    personal_info['age'] = get_user_input("Age", default="30", input_type=int, required=False)
    personal_info['gender'] = get_user_input("Gender", default="Prefer not to say", required=False)
    personal_info['nationality'] = get_user_input("Nationality", default="", required=False)
    
    return personal_info

def collect_address_info():
    """Collect address information"""
    print("\n🏠 Address Information")
    print("=" * 30)
    
    address_info = {}
    
    address_info['address'] = get_user_input("Street Address", required=False)
    address_info['city'] = get_user_input("City", required=False)
    address_info['state'] = get_user_input("State/Province", required=False)
    address_info['country'] = get_user_input("Country", required=False)
    address_info['postcode'] = get_user_input("Postal Code", required=False)
    
    return address_info

def collect_job_preferences():
    """Collect job search preferences"""
    print("\n🎯 Job Search Preferences")
    print("=" * 30)
    
    job_prefs = {}
    
    # Job titles
    default_titles = ['Software Engineer', 'Developer', 'Engineer']
    job_prefs['job_titles'] = get_list_input("What job titles are you looking for?", default_titles)
    
    # Locations
    default_locations = ['Remote', 'London', 'UK']
    job_prefs['locations'] = get_list_input("What locations are you interested in?", default_locations)
    
    # Salary
    job_prefs['salary_min'] = get_user_input("Minimum salary (annual)", default="50000", input_type=int, required=False)
    job_prefs['salary_max'] = get_user_input("Maximum salary (annual)", default="100000", input_type=int, required=False)
    job_prefs['currency'] = get_user_input("Currency", default="GBP", required=False)
    
    # Employment types
    default_employment = ['Full-time', 'Part-time', 'Contract']
    job_prefs['employment_types'] = get_list_input("What employment types are you interested in?", default_employment)
    
    # Skills
    default_skills = ['Python', 'JavaScript', 'React', 'Node.js']
    job_prefs['required_skills'] = get_list_input("What are your key skills?", default_skills)
    
    # Experience levels
    default_experience = ['Junior', 'Mid-level', 'Senior']
    job_prefs['experience_levels'] = get_list_input("What experience levels are you targeting?", default_experience)
    
    # Remote work
    default_remote = ['Remote', 'Hybrid', 'On-site']
    job_prefs['remote_flexibility'] = get_list_input("What remote work options do you prefer?", default_remote)
    
    return job_prefs

def collect_document_paths():
    """Collect document file paths"""
    print("\n📄 Document Paths")
    print("=" * 30)
    print("Enter the paths to your documents. Leave blank if you don't have them yet.")
    
    doc_paths = {}
    
    doc_paths['resume_path'] = get_user_input("Resume/CV file path", required=False)
    doc_paths['cover_letter_path'] = get_user_input("Cover letter file path", required=False)
    doc_paths['photo_path'] = get_user_input("Profile photo file path", required=False)
    
    return doc_paths

def collect_automation_settings():
    """Collect automation behavior settings"""
    print("\n⚙️ Automation Settings")
    print("=" * 30)
    
    settings = {}
    
    settings['max_jobs_to_find'] = get_user_input("Maximum jobs to find per session", default="5", input_type=int, required=False)
    settings['max_applications_per_day'] = get_user_input("Maximum applications per day", default="10", input_type=int, required=False)
    settings['min_job_match_score'] = get_user_input("Minimum job match score (0-100)", default="70", input_type=int, required=False)
    settings['auto_apply'] = get_user_input("Automatically submit applications? (y/N)", default="False", input_type=bool, required=False)
    settings['headless'] = get_user_input("Run browser in background? (y/N)", default="False", input_type=bool, required=False)
    
    return settings

def validate_config(config):
    """Validate the configuration"""
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
    
    # Check job match score
    if config.get('min_job_match_score'):
        if not (0 <= config['min_job_match_score'] <= 100):
            errors.append("Job match score must be between 0 and 100")
    
    return errors

def save_config(config, filename="user_config.json"):
    """Save configuration to JSON file"""
    try:
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"✅ Configuration saved to {filename}")
        return True
    except Exception as e:
        print(f"❌ Error saving configuration: {e}")
        return False

def load_config(filename="user_config.json"):
    """Load configuration from JSON file"""
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                return json.load(f)
        return None
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return None

def show_config_summary(config):
    """Show a summary of the configuration"""
    print("\n📋 Configuration Summary")
    print("=" * 40)
    
    print(f"👤 Name: {config.get('first_name', 'N/A')} {config.get('last_name', 'N/A')}")
    print(f"📧 Email: {config.get('email', 'N/A')}")
    print(f"📱 Phone: {config.get('phone', 'N/A')}")
    
    if config.get('locations'):
        print(f"📍 Locations: {', '.join(config['locations'][:3])}")
    
    if config.get('job_titles'):
        print(f"🎯 Job Titles: {', '.join(config['job_titles'][:3])}")
    
    if config.get('salary_min'):
        currency = config.get('currency', 'GBP')
        print(f"💰 Salary: {currency} {config['salary_min']:,}+")
    
    if config.get('required_skills'):
        print(f"🔧 Skills: {', '.join(config['required_skills'][:3])}")
    
    print(f"⚙️ Max Jobs: {config.get('max_jobs_to_find', 'N/A')}")
    print(f"🤖 Auto Apply: {config.get('auto_apply', 'N/A')}")

def setup_process():
    """Run the complete setup process"""
    print("🆕 Welcome to the Job Application Automation Engine!")
    print("This setup will help you configure your preferences.")
    
    # Check if config already exists
    existing_config = load_config()
    if existing_config:
        print(f"\n📁 Found existing configuration file: user_config.json")
        overwrite = input("Do you want to overwrite it? (y/N): ").strip().lower()
        if overwrite not in ['y', 'yes']:
            print("✅ Keeping existing configuration.")
            return existing_config
    
    print("\n🚀 Starting setup process...")
    
    # Collect all information
    config = {}
    
    # Personal info
    config.update(collect_personal_info())
    
    # Address info
    config.update(collect_address_info())
    
    # Job preferences
    config.update(collect_job_preferences())
    
    # Document paths
    config.update(collect_document_paths())
    
    # Automation settings
    config.update(collect_automation_settings())
    
    # Validate configuration
    print("\n🔍 Validating configuration...")
    errors = validate_config(config)
    
    if errors:
        print(f"\n❌ Found {len(errors)} configuration errors:")
        for error in errors:
            print(f"   - {error}")
        
        fix_errors = input("\nDo you want to fix these errors now? (y/N): ").strip().lower()
        if fix_errors in ['y', 'yes']:
            print("Please restart the setup process to fix the errors.")
            return None
    
    # Show summary
    show_config_summary(config)
    
    # Save configuration
    save_success = input("\nSave this configuration? (Y/n): ").strip().lower()
    if save_success not in ['n', 'no']:
        if save_config(config):
            print("🎉 Setup completed successfully!")
            return config
        else:
            print("❌ Failed to save configuration.")
            return None
    else:
        print("❌ Configuration not saved.")
        return None
