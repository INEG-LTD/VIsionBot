# 🚀 Job Application Automation Engine - Setup Guide

This guide explains how to set up and use the Job Application Automation Engine with the new user-friendly setup process.

## 🆕 First Time Setup

### 1. Run the Setup Process

```bash
python3 main.py
```

The first time you run `main.py`, it will detect that no configuration exists and automatically start the setup process.

### 2. Complete the Setup Wizard

The setup process will guide you through:

#### 👤 Personal Information
- **First Name** (required)
- **Last Name** (required)  
- **Email Address** (required)
- **Phone Number** (required)
- **Age** (optional)
- **Gender** (optional)
- **Nationality** (optional)

#### 🏠 Address Information
- **Street Address** (optional)
- **City** (optional)
- **State/Province** (optional)
- **Country** (optional)
- **Postal Code** (optional)

#### 🎯 Job Search Preferences
- **Job Titles** - What positions you're looking for
- **Locations** - Where you want to work
- **Salary Range** - Your salary expectations
- **Employment Types** - Full-time, part-time, contract, etc.
- **Skills** - Your technical skills and expertise
- **Experience Levels** - Junior, mid-level, senior, etc.
- **Remote Work** - Remote, hybrid, or on-site preferences

#### 📄 Document Paths
- **Resume/CV** - Path to your resume file
- **Cover Letter** - Path to your cover letter
- **Profile Photo** - Path to your profile picture

#### ⚙️ Automation Settings
- **Max Jobs** - How many jobs to find per session
- **Max Applications** - Daily application limit
- **Job Match Score** - Minimum score to consider a job
- **Auto Apply** - Whether to automatically submit applications
- **Headless Mode** - Run browser in background

### 3. Save Your Configuration

After completing the setup, your configuration will be saved to `user_config.json`. This file contains all your personal preferences and can be edited manually if needed.

## 🔄 Subsequent Runs

After the first setup, `main.py` will automatically load your saved configuration and show a main menu:

```
🎯 Job Application Automation Engine
==================================================
What would you like to do?
[1] Full Job Application | Job board → Filter Acceptable Jobs → Apply to acceptable jobs (in development)
[2] Easy apply
[3] Job Form Filler
[4] Edit/Delete config
[5] Exit
```

### Option 1: Full Job Application Pipeline
**Status**: In Development  
This feature will provide a complete automated job application pipeline including:
- Job board scraping and monitoring
- AI-powered job filtering and scoring  
- Automated application submission
- Progress tracking and analytics

### Option 2: Easy Apply
Quick application to jobs you've already found. The system will:
- Fill out application forms automatically
- Upload your resume and cover letter
- Submit applications with one click

### Option 3: Job Form Filler
Fill out job application forms with your saved preferences:
- Auto-fill personal information
- Populate job-specific fields
- Handle various form layouts

### Option 4: Edit/Delete Configuration
Manage your configuration with options to:
- View current configuration
- Edit configuration manually
- Delete configuration and restart setup
- Return to main menu

### Option 5: Exit
Closes the application.

## 🔧 Configuration Management

### File Structure

- **`config.py`** - Default configuration with sensible defaults
- **`user_setup.py`** - User setup and configuration management functions
- **`user_config.json`** - Your personal preferences (created during setup)
- **`main.py`** - Main application that uses the setup module

### How Configuration Works

1. **Default Config** (`config.py`) provides comprehensive defaults for all features
2. **User Config** (`user_config.json`) contains your personal preferences
3. **Merged Config** combines both, with your preferences taking priority
4. **Automation Engine** uses the merged configuration

### Editing Configuration

#### Method 1: Manual Edit
Edit `user_config.json` directly in any text editor:

```json
{
  "first_name": "Your Name",
  "job_titles": ["Software Engineer", "Developer"],
  "salary_min": 60000
}
```

#### Method 2: Restart Setup
Delete `user_config.json` and run `main.py` again to restart the setup process.

#### Method 3: Use Integration Script
Run `python3 integrate_user_config.py` to see how your config integrates with defaults.

## 🧪 Testing Your Setup

### Test Configuration
```bash
python3 test_config.py
```

### Test Integration
```bash
python3 integrate_user_config.py
```

### View Configuration Summary
```bash
python3 config.py
```

## 📁 Example Configuration

Here's what your `user_config.json` might look like:

```json
{
  "first_name": "chris",
  "last_name": "ineg",
  "email": "eromoseleinegbe@gmail.com",
  "phone": "07365260995",
  "age": 23,
  "gender": "Male",
  "nationality": "British",
  "address": "36 Slade Baker Way",
  "city": "Bristol",
  "state": "England",
  "country": "United Kingdom",
  "postcode": "BS161QT",
  "job_titles": ["ios engineer"],
  "locations": ["Remote", "London", "UK"],
  "salary_min": 50000,
  "salary_max": 100000,
  "currency": "GBP",
  "employment_types": ["Full time"],
  "required_skills": ["Swift", "Git", "Python"],
  "experience_levels": ["Mid-level"],
  "remote_flexibility": ["Remote"],
  "max_jobs_to_find": 10,
  "max_applications_per_day": 10,
  "min_job_match_score": 70,
  "auto_apply": false,
  "headless": false
}
```

## 🚨 Troubleshooting

### Common Issues

1. **Setup Crashes**
   - Check Python version (3.6+ required)
   - Ensure all dependencies are installed
   - Check file permissions

2. **Configuration Not Saved**
   - Verify write permissions in current directory
   - Check disk space
   - Look for error messages in console

3. **Invalid Configuration**
   - Run `python3 test_config.py` to validate
   - Check required fields are filled
   - Verify salary range is logical

4. **Automation Won't Start**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check that `user_config.json` exists and is valid JSON
   - Verify document paths point to actual files

### Getting Help

- Check console output for specific error messages
- Run validation scripts to identify issues
- Review the configuration file format
- Ensure all required fields are completed

## 🎯 Best Practices

### Setup
1. **Complete all required fields** - First name, last name, email, phone
2. **Use realistic values** - Set achievable salary ranges and experience levels
3. **Test your configuration** - Use the test scripts before running automation
4. **Keep backups** - Save a copy of working configurations

### Maintenance
1. **Update regularly** - Keep your skills and preferences current
2. **Review settings** - Periodically check automation behavior
3. **Backup configs** - Save working configurations before major changes
4. **Validate changes** - Test configuration after modifications

## 🔗 Integration with Existing System

The new setup process integrates seamlessly with the existing automation engine:

- **Backward Compatible** - Works with existing `config.py`
- **Seamless Integration** - User preferences override defaults automatically
- **No Code Changes** - Existing automation code works unchanged
- **Flexible** - Can use just user config or merged config

## 📚 Additional Resources

- **`CONFIG_README.md`** - Detailed configuration documentation
- **`user_setup.py`** - User setup and configuration management module
- **`example_usage.py`** - Examples of using the configuration system
- **`run_with_config.py`** - Alternative way to run with configuration
- **`test_config.py`** - Configuration validation and testing
- **`test_user_setup.py`** - Testing the user setup module

---

**🎉 You're all set! Run `python3 main.py` to get started with your personalized job application automation.**
