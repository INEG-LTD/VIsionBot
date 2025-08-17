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
import time
from InquirerPy.validator import EmptyInputValidator, NumberValidator
from rich import print as rprint
from InquirerPy.base.control import Choice
from InquirerPy import inquirer

from terminal import clear_screen_preserve_banner, banner_lines

def get_user_input(prompt, instruction="", multi_inputs=False, default="", mandatory=False, input_type=str) -> str | list[str] | int | float | bool:
    """Get user input with validation"""
    message = f"{prompt}" if not default else f"{prompt} (default: {default})"
    
    if input_type == bool:
        result = inquirer.confirm(
            message=message,
            default=False if default.lower() == "false" else True,
            long_instruction=instruction if instruction else "",
        ).execute()
    elif input_type in (int, float):
        if multi_inputs:
            result = inquirer.text(
                message=f"{'(required) ' if mandatory else ''}{message} (comma separated)",
                mandatory=mandatory,
                long_instruction=instruction if instruction else "",
                mandatory_message=f"This field is mandatory. Please enter a value.",
                validate=EmptyInputValidator("Input should not be empty") if mandatory else None,
            ).execute()
            if result:
                values = [input_type(x.strip()) for x in result.split(",")]
                return values
            return []
        
        # Handle empty default for numeric inputs
        numeric_default = None
        if default and str(default).strip():
            try:
                numeric_default = int(default) if input_type == int else float(default)
            except (ValueError, TypeError):
                numeric_default = None
        
        result = inquirer.number(
            message=f"{'(required) ' if mandatory else ''}{message}",
            default=numeric_default,
            float_allowed=(input_type == float),
            long_instruction=instruction if instruction else "",
            mandatory=mandatory,
            mandatory_message=f"This field is mandatory. Please enter a value.",
            validate=NumberValidator("Input should be a number") if mandatory else None
        ).execute()
    else:
        if multi_inputs:
            result = inquirer.text(
                message=f"{'(required) ' if mandatory else ''}{message} (comma separated)",
                long_instruction=instruction if instruction else "",
                mandatory=mandatory,
                mandatory_message=f"This field is mandatory. Please enter a value.",
                validate=EmptyInputValidator("Input should not be empty") if mandatory else None,
            ).execute()
            if result:
                values = [x.strip() for x in result.split(",")]
                return values
            return []
        
        result = inquirer.text(
            message=f"{'(required) ' if mandatory else ''}{message}",
            default=default,
            long_instruction=instruction if instruction else "",
            mandatory=mandatory,
            mandatory_message=f"This field is mandatory. Please enter a value.",
            validate=EmptyInputValidator("Input should not be empty") if mandatory else None,
        ).execute()
        
    if input_type == int:
        result = int(result)
    elif input_type == float:
        result = float(result)
    elif input_type == list[str]:
        result = [x.strip() for x in result.split(",")]

    return result if result or not mandatory else default

def get_checkbox_input(prompt, options, default=[]) -> list[str]:
    """Get a list of items from user"""
    if default:
        use_default = inquirer.confirm(
            message=f"Use default values? ({', '.join(default)})",
            default=False
        ).execute()
        if use_default:
            return default
    
    return inquirer.checkbox(
        message=prompt,
        choices=options,
        default=default
    ).execute()
    
def get_select_input(prompt, options, default=None) -> str:
    """Get a single item from user"""
    return inquirer.select(
        message=prompt,
        choices=options,
        default=default
    ).execute()

def collect_personal_info():
    """Collect basic personal information"""
    rprint("\n[bold underline]👤 Personal Information[/bold underline]\n")
    
    personal_info = {}
    
    personal_info['first_name'] = get_user_input("First Name", mandatory=True)
    personal_info['last_name'] = get_user_input("Last Name", mandatory=True)
    personal_info['email'] = get_user_input("Email Address", mandatory=True)
    personal_info['phone'] = get_user_input("Phone Number", input_type=int, mandatory=True)
    personal_info['age'] = get_user_input("Age", input_type=int, mandatory=False)
    personal_info['gender'] = get_select_input("Gender", options=["Male", "Female", "Prefer not to say"], default="Prefer not to say")
    personal_info['nationality'] = get_user_input("Nationality", default="", mandatory=False)
    
    return personal_info

def collect_address_info():
    """Collect address information"""
    rprint("\n[bold underline]🏠 Address Information[/bold underline]\n")
    
    address_info = {}
    
    address_info['address'] = get_user_input("Street Address", mandatory=False)
    address_info['city'] = get_user_input("City", mandatory=False)
    address_info['state'] = get_user_input("State/Province", mandatory=False)
    address_info['country'] = get_user_input("Country", mandatory=False)
    address_info['postcode'] = get_user_input("Postal Code", mandatory=False)
    
    return address_info

def collect_job_preferences():
    """Collect job search preferences"""
    rprint("\n[bold underline]🎯 Job Search Preferences[/bold underline]\n")
    
    job_prefs = {}
    
    # Job titles
    job_prefs['job_titles'] = get_user_input(
        prompt="What job titles are you looking for?",
        multi_inputs=True,
        input_type=list[str],
        mandatory=True
    )
    
    # Locations
    job_prefs['locations'] = get_user_input(
        prompt="What locations are you interested in?",
        multi_inputs=True,
        input_type=list[str],
        mandatory=True
    )
    
    # Salary
    job_prefs['salary_min'] = get_user_input(
        prompt="Minimum salary (annual)",
        input_type=int,
        default="50000",
        mandatory=False
    )
    job_prefs['salary_max'] = get_user_input(
        prompt="Maximum salary (annual)",
        input_type=int,
        default="100000",
        mandatory=False
    )
    job_prefs['currency'] = get_user_input(
        prompt="Currency",
        input_type=str,
        default="GBP",
        mandatory=False
    )
    
    # Employment types
    job_prefs['employment_types'] = get_select_input(
        prompt="What employment types are you interested in?",
        options=["Full-time", "Part-time", "Contract"],
        default="Full-time"
    )
    
    # Skills
    job_prefs['required_skills'] = get_user_input("What are your key skills?", multi_inputs=True, input_type=list[str], mandatory=True)
    
    # Experience levels
    job_prefs['experience_levels'] = get_select_input(
        prompt="What experience levels are you targeting?",
        options=["Junior", "Mid-level", "Senior"],
        default="Junior"
    )
    
    # Remote work
    job_prefs['remote_flexibility'] = get_select_input(
        prompt="What remote work options do you prefer?",
        options=["Remote", "Hybrid", "On-site"],
        default="Remote"
    )
    
    return job_prefs

def collect_document_paths():
    """Collect document file paths"""
    rprint("\n[bold underline]📄 Document Paths[/bold underline]\n")
    rprint("Enter the paths to your documents. Leave blank if you don't have them yet.\n")
    
    doc_paths = {}
    
    doc_paths['resume_path'] = get_user_input("Resume/CV file path", mandatory=False)
    doc_paths['cover_letter_path'] = get_user_input("Cover letter file path", mandatory=False)
    doc_paths['photo_path'] = get_user_input("Profile photo file path", mandatory=False)
    
    return doc_paths

def collect_automation_settings():
    """Collect automation behavior settings"""
    rprint("\n[bold underline]⚙️ Automation Settings[/bold underline]\n")
    
    settings = {}
    
    settings['max_jobs_to_find'] = get_user_input("Maximum jobs to find per session", instruction="This is the maximum number of jobs you want to find per session from an individual job board.", default="5", input_type=int, mandatory=False)
    settings['max_applications_per_day'] = get_user_input("Maximum applications per day", instruction="This is the maximum number of applications you want to submit per day. This is a safety measure to prevent you from submitting too many applications and getting flagged as spam.", default="10", input_type=int, mandatory=False)
    settings['min_job_match_score'] = get_user_input("Minimum job match score (0-100)", instruction="Each job is scored based on how well it matches the preferences you set.\nThis is the minimum score you want to achieve for a job to be considered a match. This is a safety measure to prevent you from applying to jobs that are not a good match for you.", default="70", input_type=int, mandatory=False)
    settings['auto_apply'] = get_user_input("Automatically submit applications? (y/N)", instruction="This will allow you to apply to jobs without having to manually click the apply button..", default="False", input_type=bool, mandatory=False)
    settings['headless'] = get_user_input("Run browser in background? (y/N)", instruction="This will run the browser in background.", default="False", input_type=bool, mandatory=False)
    
    return settings

def validate_config(config):
    """Validate the configuration"""
    errors = []
    
    # Check mandatory fields
    mandatory_fields = ['first_name', 'last_name', 'email', 'phone']
    for field in mandatory_fields:
        if not config.get(field):
            errors.append(f"Missing mandatory field: {field}")
    
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
        
        clear_screen_preserve_banner(banner_lines)
        print(f"✅ Configuration saved to {filename}")
        return True
    except Exception as e:
        print(f"❌ Error saving configuration: {e}")
        return False

def load_config(filename="user_config.json") -> dict | None:
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
    
    rprint(f"[bold]👤 Name:[/bold] {config.get('first_name', 'N/A')} {config.get('last_name', 'N/A')}")
    rprint(f"[bold]📧 Email:[/bold] {config.get('email', 'N/A')}")
    rprint(f"[bold]📱 Phone:[/bold] {config.get('phone', 'N/A')}")
    rprint(f"[bold]📍 Locations:[/bold] {', '.join(config.get('locations', ['N/A']))}")
    rprint(f"[bold]🎯 Job Titles:[/bold] {', '.join(config.get('job_titles', ['N/A']))}")
    rprint(f"[bold]💰 Salary Range:[/bold] {config.get('currency', 'GBP')} {config.get('salary_min', 'N/A')} - {config.get('salary_max', 'N/A')}")
    rprint(f"[bold]🔧 Required Skills:[/bold] {', '.join(config.get('required_skills', ['N/A']))}")
    rprint(f"[bold]👔 Employment Type:[/bold] {config.get('employment_type', 'N/A')}")
    rprint(f"[bold]📈 Experience Level:[/bold] {config.get('experience_levels', 'N/A')}")
    rprint(f"[bold]🏠 Remote Work:[/bold] {config.get('remote_flexibility', 'N/A')}")
    rprint(f"[bold]📄 Resume Path:[/bold] {config.get('resume_path', 'N/A')}")
    rprint(f"[bold]✉️ Cover Letter Path:[/bold] {config.get('cover_letter_path', 'N/A')}")
    rprint(f"[bold]🖼️ Photo Path:[/bold] {config.get('photo_path', 'N/A')}")
    rprint(f"[bold]⚙️ Max Jobs Per Session:[/bold] {config.get('max_jobs_to_find', 'N/A')}")
    rprint(f"[bold]📊 Max Applications/Day:[/bold] {config.get('max_applications_per_day', 'N/A')}")
    rprint(f"[bold]🎯 Min Match Score:[/bold] {config.get('min_job_match_score', 'N/A')}")
    rprint(f"[bold]🤖 Auto Apply:[/bold] {config.get('auto_apply', 'N/A')}")
    rprint(f"[bold]🖥️ Headless Mode:[/bold] {config.get('headless', 'N/A')}")
    print("\n")

def setup_process():
    """Run the complete setup process"""
    clear_screen_preserve_banner(banner_lines)
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
        
    # Collect all information
    config = {}
    
    # Personal info
    config.update(collect_personal_info())
    clear_screen_preserve_banner(banner_lines)
    
    # Address info
    config.update(collect_address_info())
    clear_screen_preserve_banner(banner_lines)
    
    # Job preferences
    config.update(collect_job_preferences())
    clear_screen_preserve_banner(banner_lines)
    
    # Document paths
    config.update(collect_document_paths())
    clear_screen_preserve_banner(banner_lines)
    
    # Automation settings
    config.update(collect_automation_settings())
    clear_screen_preserve_banner(banner_lines)
    
    # Validate configuration
    errors = validate_config(config)

    if errors:
        print(f"\n❌ Found {len(errors)} configuration errors:")
        for error in errors:
            print(f"   - {error}")
        
        fix_errors = get_user_input("\nDo you want to fix these errors now? (y/N): ", input_type=bool, default="False", mandatory=True)
        if fix_errors:
            clear_screen_preserve_banner(banner_lines)
            setup_process()
            return None
    
    # Show summary
    show_config_summary(config)
    
    # Save configuration
    save_success = get_user_input("Save this configuration? (Y/n): ", input_type=bool, default="True", mandatory=True)
    if save_success:
        if save_config(config):
            print("🎉 Setup completed successfully!\n")
            return config
        else:
            print("❌ Failed to save configuration.")
            return None
    else:
        print("❌ Configuration not saved.")
        return None
