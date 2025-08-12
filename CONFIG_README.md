# Configuration Guide for Job Application Automation Engine

This guide explains how to customize the Job Application Automation Engine using the `config.py` file.

## 🚀 Quick Start

1. **Edit the config file**: Open `config.py` and modify the values to match your preferences
2. **Run the example**: Execute `python example_usage.py` to test your configuration
3. **Start automation**: Use your configured preferences with the main automation engine

## 📁 Configuration Structure

The `config.py` file is organized into logical sections for easy customization:

### 👤 Personal Information
```python
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
```

### 🏠 Address Information
```python
ADDRESS_INFO = {
    'address': '123 Main St, Anytown, USA',
    'city': 'Bristol',
    'state': 'England',
    'country': 'United Kingdom',
    'postcode': 'BS1 1AA',
}
```

### 📄 Document Paths
```python
DOCUMENT_PATHS = {
    'resume_path': 'cv.pdf',
    'cover_letter_path': 'cover_letter.pdf',
    'photo_path': 'photo.png',
}
```

### 🎯 Job Search Preferences
```python
JOB_TITLES = [
    'iOS Engineer',
    'Senior iOS Engineer', 
    'Mobile Developer',
    'Software Engineer',
]

LOCATIONS = [
    'London',
    'UK', 
    'United Kingdom',
    'Remote',
]

SALARY_PREFERENCES = {
    'salary_min': 80000,
    'salary_max': 150000,
    'currency': 'GBP',
    'negotiable': True,
}
```

### 🔧 Skills and Experience
```python
REQUIRED_SKILLS = [
    'Swift',
    'iOS',
    'SwiftUI',
    'Python',
    'JavaScript',
]

EXPERIENCE_LEVELS = [
    'Junior',
    'Mid-level',
    'Senior',
    'Lead',
]
```

## ⚙️ Advanced Settings

### Browser Configuration
```python
BROWSER_SETTINGS = {
    'headless': False,  # Set to True to run in background
    'max_restarts': 3,
    'timeout': 30000,
    'screenshot_on_error': True,
}
```

### AI Model Settings
```python
AI_SETTINGS = {
    'model_name': 'gemini-2.5-flash',
    'max_retries': 3,
    'temperature': 0.7,
    'max_tokens': 1000,
}
```

### Application Behavior
```python
APPLICATION_SETTINGS = {
    'max_jobs_to_find': 5,
    'max_applications_per_day': 10,
    'min_job_match_score': 70,
    'auto_apply': False,  # Set to True for automatic submission
    'save_applications': True,
}
```

## 🔄 How to Use

### 1. Basic Usage
```python
from config import get_all_preferences
from find_jobs import FindJobsBot

# Load your preferences
preferences = get_all_preferences()

# Initialize the bot with your preferences
bot = FindJobsBot(headless=False, preferences=preferences)
```

### 2. Override Specific Settings
```python
# Load default preferences
preferences = get_all_preferences()

# Override specific values
preferences['max_jobs_to_find'] = 10
preferences['headless'] = True

# Use modified preferences
bot = FindJobsBot(preferences=preferences)
```

### 3. Validate Configuration
```python
from config import validate_config

errors = validate_config()
if errors:
    print("Configuration errors found:")
    for error in errors:
        print(f"  - {error}")
else:
    print("Configuration is valid!")
```

## 📝 Customization Examples

### Add New Job Titles
```python
JOB_TITLES = [
    'iOS Engineer',
    'Senior iOS Engineer', 
    'Mobile Developer',
    'Software Engineer',
    'Your Custom Title Here',  # Add your own
]
```

### Update Salary Range
```python
SALARY_PREFERENCES = {
    'salary_min': 100000,  # Change minimum
    'salary_max': 200000,  # Change maximum
    'currency': 'USD',      # Change currency
    'negotiable': True,
}
```

### Add New Skills
```python
REQUIRED_SKILLS = [
    'Swift',
    'iOS',
    'SwiftUI',
    'Combine',
    'Your New Skill',  # Add your own
]
```

### Configure Remote Work Preferences
```python
REMOTE_FLEXIBILITY = [
    'Hybrid',
    'Remote',
    'On-site',
    'Flexible',
    'Your Preference',  # Add your own
]
```

## 🚨 Important Notes

### Required Fields
The following fields are required and must be filled:
- `first_name`
- `last_name` 
- `email`
- `phone`

### Document Paths
Make sure the document paths point to actual files:
- Resume should be a PDF or common format
- Cover letter should be a PDF or text file
- Photo should be a common image format (PNG, JPG)

### Salary Range
- `salary_min` cannot be greater than `salary_max`
- Use appropriate currency codes (GBP, USD, EUR, etc.)

### AI Settings
- `temperature` must be between 0.0 and 1.0
- Higher values = more creative responses
- Lower values = more focused responses

## 🧪 Testing Your Configuration

1. **Run the example script**:
   ```bash
   python example_usage.py
   ```

2. **Check configuration summary**:
   ```bash
   python config.py
   ```

3. **Validate configuration**:
   ```python
   from config import validate_config
   errors = validate_config()
   print(f"Found {len(errors)} errors")
   ```

## 🔧 Troubleshooting

### Common Issues

1. **Document not found errors**:
   - Check that resume, cover letter, and photo files exist
   - Use absolute paths if needed: `/full/path/to/document.pdf`

2. **Configuration validation errors**:
   - Ensure required fields are filled
   - Check salary range is logical
   - Verify AI settings are within valid ranges

3. **Import errors**:
   - Make sure `config.py` is in the same directory
   - Check that all dependencies are installed

### Getting Help

- Run `python config.py` to see your current configuration
- Use `python example_usage.py` to test your setup
- Check the console output for specific error messages

## 📚 Advanced Customization

### Custom CSS Selectors
For specific job sites, you can add custom selectors:
```python
WEBSITE_SETTINGS = {
    'custom_selectors': {
        'linkedin': {
            'job_title': 'h1.job-title',
            'apply_button': 'button.apply-button',
        },
        'indeed': {
            'job_title': 'h1.jobsearch-JobInfoHeader-title',
            'apply_button': 'button[data-indeed-apply-button]',
        },
    },
}
```

### Rate Limiting
Control how fast the bot makes requests:
```python
WEBSITE_SETTINGS = {
    'rate_limiting': {
        'requests_per_minute': 20,  # Reduce for slower sites
        'delay_between_requests': 3,  # Increase delay
    },
}
```

### Notification Settings
Configure email notifications:
```python
NOTIFICATION_SETTINGS = {
    'email_notifications': True,
    'email_smtp_server': 'smtp.gmail.com',
    'email_username': 'your-email@gmail.com',
    'email_password': 'your-app-password',  # Use app password, not regular password
}
```

## 🎯 Best Practices

1. **Start with defaults**: Use the provided defaults and modify gradually
2. **Test incrementally**: Make small changes and test frequently
3. **Backup your config**: Keep a backup of working configurations
4. **Use descriptive values**: Make your preferences specific and clear
5. **Regular updates**: Review and update your preferences periodically

## 📞 Support

If you encounter issues:
1. Check the console output for error messages
2. Validate your configuration using `validate_config()`
3. Ensure all required files and dependencies are present
4. Review the example usage script for reference

---

**Happy job hunting! 🚀**
