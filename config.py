#!/usr/bin/env python3
"""
Configuration file for Job Application Automation Engine

This file contains all user preferences and settings that can be modified
to customize the behavior of the automation engine.
"""

import os
from typing import Dict, Any, List

# ============================================================================
# PERSONAL INFORMATION
# ============================================================================

# Basic personal details
PERSONAL_INFO = {
    'first_name': 'John',
    'last_name': 'Doe',
    'email': 'john.doe@example.com',
    'phone': '+1234567890',
    'age': 30,
    'gender': 'Male',
    'date_of_birth': '1990-01-01',
    'nationality': 'British',
    'marital_status': 'Single',
}

# Address information
ADDRESS_INFO = {
    'address': '123 Main St, Anytown, USA',
    'city': 'Bristol',
    'state': 'England',
    'county/region': 'Gloucestershire',
    'country': 'United Kingdom',
    'postcode': 'BS1 1AA',
    'zip': '12345',  # US format alternative
}

# Additional personal details
ADDITIONAL_INFO = {
    'driving_license': 'Yes',
    'car_ownership': 'Yes',
    'car_model': 'Toyota Corolla',
    'car_year': 2020,
    'disability': 'No',
    'disability_description': 'No',
    'disability_type': 'No',
    'religion': 'Christian',
    'relationship_status': 'Single',
    'relationship_status_description': 'No',
    'relationship_status_type': 'No',
    'ethnicity': 'White',
    'ethnicity_description': 'No',
    'ethnicity_type': 'No',
    'pregnant': 'No',
}

# ============================================================================
# DOCUMENT PATHS
# ============================================================================

# File paths for documents (relative to project directory or absolute paths)
DOCUMENT_PATHS = {
    'resume_path': 'cv.pdf',
    'cover_letter_path': 'cover_letter.pdf',
    'photo_path': 'photo.png',
}

# ============================================================================
# JOB SEARCH PREFERENCES
# ============================================================================

# Job titles to search for
JOB_TITLES = [
    'iOS Engineer',
    'Senior iOS Engineer', 
    'Mobile Developer',
    'Software Engineer',
    'Full Stack Developer',
    'Backend Developer',
    'Frontend Developer',
    'DevOps Engineer',
    'Data Scientist',
    'Product Manager',
]

# Preferred locations
LOCATIONS = [
    'London',
    'UK', 
    'United Kingdom',
    'Remote',
    'Bristol',
    'Manchester',
    'Edinburgh',
]

# Salary expectations
SALARY_PREFERENCES = {
    'salary_min': 80000,
    'salary_max': 150000,
    'currency': 'GBP',
    'negotiable': True,
}

# Employment types
EMPLOYMENT_TYPES = [
    'Full-time',
    'Part-time',
    'Contract',
    'Freelance',
    'Internship',
]

# Required skills and technologies
REQUIRED_SKILLS = [
    'Swift',
    'iOS',
    'SwiftUI',
    'Combine',
    'Mobile Development',
    'Python',
    'JavaScript',
    'React',
    'Node.js',
    'Docker',
    'AWS',
    'Git',
]

# Experience levels
EXPERIENCE_LEVELS = [
    'Junior',
    'Mid-level',
    'Senior',
    'Lead',
    'Principal',
    'Architect',
]

# Remote work preferences
REMOTE_FLEXIBILITY = [
    'Hybrid',
    'Remote',
    'On-site',
    'Flexible',
]

# Desired benefits
DESIRED_BENEFITS = [
    'Health Insurance',
    'Stock Options',
    'Learning Budget',
    'Flexible Working Hours',
    'Remote Work',
    'Professional Development',
    'Pension',
    'Life Insurance',
    'Dental Insurance',
    'Vision Insurance',
    'Gym Membership',
    'Free Lunch',
    'Transportation Allowance',
]

# Keywords to exclude from job searches
EXCLUDE_KEYWORDS = [
    'unpaid',
    'internship',
    'junior',
    'volunteer',
    'commission only',
    'no experience',
    'entry level',
    'trainee',
]

# ============================================================================
# VISA AND WORK AUTHORIZATION
# ============================================================================

VISA_PREFERENCES = {
    'visa_sponsorship_required': False,
    'work_authorization': 'UK Citizen',  # or 'Work Visa', 'Student Visa', etc.
    'willing_to_relocate': True,
    'relocation_assistance_required': False,
}

# ============================================================================
# APPLICATION BEHAVIOR SETTINGS
# ============================================================================

APPLICATION_SETTINGS = {
    'max_jobs_to_find': 5,
    'max_applications_per_day': 10,
    'min_job_match_score': 70,  # Percentage (0-100)
    'auto_apply': False,  # Whether to automatically submit applications
    'save_applications': True,  # Save application data locally
    'follow_up_emails': True,  # Send follow-up emails after applying
}

# ============================================================================
# BROWSER SETTINGS
# ============================================================================

BROWSER_SETTINGS = {
    'headless': False,  # Run browser in background
    'max_restarts': 3,  # Maximum browser restart attempts
    'timeout': 30000,  # Page load timeout in milliseconds
    'wait_for_dom': True,  # Wait for DOM content to load
    'screenshot_on_error': True,  # Take screenshots on errors
}

# ============================================================================
# AI MODEL SETTINGS
# ============================================================================

AI_SETTINGS = {
    'model_name': 'gemini-2.5-flash',  # AI model to use for analysis
    'max_retries': 3,  # Maximum AI API call retries
    'temperature': 0.7,  # AI response creativity (0.0-1.0)
    'max_tokens': 1000,  # Maximum tokens in AI response
}

# ============================================================================
# LOGGING AND DEBUGGING
# ============================================================================

LOGGING_SETTINGS = {
    'log_level': 'INFO',  # DEBUG, INFO, WARNING, ERROR
    'log_to_file': True,
    'log_file_path': 'job_automation.log',
    'verbose_output': True,
    'save_screenshots': True,
    'screenshot_directory': 'screenshots',
}

# ============================================================================
# WEBSITE SPECIFIC SETTINGS
# ============================================================================

WEBSITE_SETTINGS = {
    'job_sites': [
        'LinkedIn',
        'Indeed',
        'Glassdoor',
        'Monster',
        'ZipRecruiter',
        'AngelList',
        'Stack Overflow Jobs',
        'GitHub Jobs',
    ],
    'custom_selectors': {
        # Add custom CSS selectors for specific websites if needed
        'linkedin': {},
        'indeed': {},
        'glassdoor': {},
    },
    'rate_limiting': {
        'requests_per_minute': 30,
        'delay_between_requests': 2,  # seconds
    },
}

# ============================================================================
# NOTIFICATION SETTINGS
# ============================================================================

NOTIFICATION_SETTINGS = {
    'email_notifications': True,
    'email_smtp_server': 'smtp.gmail.com',
    'email_smtp_port': 587,
    'email_username': 'your-email@gmail.com',
    'email_password': 'your-app-password',
    'desktop_notifications': True,
    'sound_notifications': False,
}

# ============================================================================
# CONFIGURATION FUNCTIONS
# ============================================================================

def get_all_preferences() -> Dict[str, Any]:
    """
    Combine all configuration sections into a single preferences dictionary.
    This is the main function used by the automation engine.
    """
    preferences = {}
    
    # Merge all configuration sections
    sections = [
        PERSONAL_INFO,
        ADDRESS_INFO,
        ADDITIONAL_INFO,
        DOCUMENT_PATHS,
        SALARY_PREFERENCES,
        VISA_PREFERENCES,
        APPLICATION_SETTINGS,
        BROWSER_SETTINGS,
        AI_SETTINGS,
        LOGGING_SETTINGS,
        WEBSITE_SETTINGS,
        NOTIFICATION_SETTINGS,
    ]
    
    for section in sections:
        preferences.update(section)
    
    # Add list-based preferences
    preferences.update({
        'job_titles': JOB_TITLES,
        'locations': LOCATIONS,
        'employment_types': EMPLOYMENT_TYPES,
        'required_skills': REQUIRED_SKILLS,
        'experience_levels': EXPERIENCE_LEVELS,
        'remote_flexibility': REMOTE_FLEXIBILITY,
        'desired_benefits': DESIRED_BENEFITS,
        'exclude_keywords': EXCLUDE_KEYWORDS,
    })
    
    return preferences

def validate_config() -> List[str]:
    """
    Validate the configuration and return any errors found.
    """
    errors = []
    
    # Check required fields
    required_fields = ['first_name', 'last_name', 'email', 'phone']
    for field in required_fields:
        if not PERSONAL_INFO.get(field):
            errors.append(f"Missing required field: {field}")
    
    # Check document paths
    for doc_type, path in DOCUMENT_PATHS.items():
        if path and not os.path.exists(path):
            errors.append(f"Document not found: {doc_type} at {path}")
    
    # Check salary range
    if SALARY_PREFERENCES['salary_min'] > SALARY_PREFERENCES['salary_max']:
        errors.append("Minimum salary cannot be greater than maximum salary")
    
    # Check AI settings
    if AI_SETTINGS['temperature'] < 0 or AI_SETTINGS['temperature'] > 1:
        errors.append("AI temperature must be between 0.0 and 1.0")
    
    return errors

def print_config_summary():
    """
    Print a summary of the current configuration.
    """
    print("🔧 Job Application Automation Engine Configuration")
    print("=" * 60)
    
    print(f"👤 Personal: {PERSONAL_INFO['first_name']} {PERSONAL_INFO['last_name']}")
    print(f"📧 Email: {PERSONAL_INFO['email']}")
    print(f"📱 Phone: {PERSONAL_INFO['phone']}")
    print(f"📍 Location: {', '.join(LOCATIONS[:3])}")
    
    print(f"\n🎯 Job Titles: {', '.join(JOB_TITLES[:5])}")
    print(f"💰 Salary Range: {SALARY_PREFERENCES['currency']} {SALARY_PREFERENCES['salary_min']:,} - {SALARY_PREFERENCES['salary_max']:,}")
    print(f"🔧 Skills: {', '.join(REQUIRED_SKILLS[:5])}")
    
    print(f"\n⚙️ Settings:")
    print(f"   - Max Jobs: {APPLICATION_SETTINGS['max_jobs_to_find']}")
    print(f"   - Headless: {BROWSER_SETTINGS['headless']}")
    print(f"   - Auto Apply: {APPLICATION_SETTINGS['auto_apply']}")
    
    # Validate configuration
    errors = validate_config()
    if errors:
        print(f"\n❌ Configuration Errors:")
        for error in errors:
            print(f"   - {error}")
    else:
        print(f"\n✅ Configuration is valid")

if __name__ == "__main__":
    # Print configuration summary when run directly
    print_config_summary()
