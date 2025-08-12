# Job Application Automation Engine

A comprehensive Python-based automation engine for job applications using Playwright and Gemini 2.5 Pro for intelligent job searching, evaluation, and form filling.

## Overview

This automation engine provides a complete solution for job application automation, from finding relevant job listings to filling out application forms. It combines intelligent AI-powered analysis with robust web automation to streamline the job application process.

## Core Components

### 1. FindJobsBot (`find_jobs.py`)
The main automation bot that orchestrates the entire job application process:

- **Job Site Navigation**: Automatically navigates to job search sites
- **Search Form Filling**: Uses Gemini 2.5 Pro to identify and fill search forms with user preferences
- **Job Listing Detection**: AI-powered detection of job listings using Gemini 2.5 Pro
- **Job Evaluation**: Intelligent job matching based on user preferences and job descriptions
- **Multi-tab Processing**: Opens accepted jobs in new tabs for parallel processing

### 2. JobDetailToFormTransitionHandler (`job_detail_to_form_transition_handler.py`)
Handles the transition from job detail pages to application forms:

- **Page State Detection**: Automatically detects different page types (login, verification, forms, etc.)
- **Navigation Logic**: Intelligent navigation through multi-step application processes
- **Iframe Handling**: Comprehensive iframe detection and context switching
- **User Intervention**: Detects when manual input is required and pauses for user action
- **Error Recovery**: Robust error handling with fallback mechanisms

### 3. ApplicationFiller (`fill_and_submit_job_form.py`)
Specialized form filling engine with AI-powered field detection:

- **Form Field Detection**: Uses Gemini 2.5 Pro to identify all form elements
- **Smart Field Mapping**: Maps user preferences to appropriate form fields
- **Multi-field Support**: Handles text inputs, dropdowns, radio buttons, checkboxes, and file uploads
- **Iframe Context Awareness**: Seamlessly works with forms in iframes
- **Form Submission**: Intelligent submit button detection and form completion

## Key Features

### AI-Powered Intelligence
- **Gemini 2.5 Pro Integration**: Uses Google's latest AI model for intelligent analysis
- **Smart Field Recognition**: Automatically identifies form fields and their purposes
- **Job Matching**: Evaluates job fit based on user preferences and job descriptions
- **Context Understanding**: AI understands page context and form relationships

### Robust Web Automation
- **Playwright Integration**: Modern, reliable web automation framework
- **Iframe Support**: Automatic detection and handling of forms in iframes
- **Cross-browser Compatibility**: Works with Chrome and other Chromium-based browsers
- **Error Handling**: Comprehensive error recovery and fallback mechanisms

### User Experience
- **Preference-based Automation**: Fills forms based on user-defined preferences
- **Smart Job Filtering**: Only applies to jobs that match user criteria
- **Progress Tracking**: Real-time feedback on automation progress
- **User Intervention**: Pauses when manual input is required

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd job-application-automation-engine-python
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Google AI API key**
   - Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - The code currently uses a hardcoded API key (should be moved to environment variables)

4. **Install Playwright browsers**
   ```bash
   playwright install
   ```

## Configuration

### User Preferences
Configure your job search and application preferences:

```python
preferences = {
    'job_titles': ['Software Engineer', 'Developer', 'Programmer'],
    'locations': ['London', 'Remote', 'New York'],
    'salary_min': 50000,
    'employment_types': ['Full-time', 'Contract'],
    'required_skills': ['Python', 'JavaScript', 'React'],
    'experience_levels': ['Mid', 'Senior'],
    'visa_sponsorship_required': False,
    'remote_flexibility': ['Remote', 'Hybrid'],
    'desired_benefits': ['Health Insurance', 'Stock Options'],
    'exclude_keywords': ['unpaid', 'internship', 'volunteer']
}
```

### Browser Configuration
- **Headless Mode**: Set `headless=False` for debugging and manual intervention
- **Chrome Profile**: Uses separate automation profiles to avoid conflicts
- **Iframe Detection**: Automatically enabled for comprehensive form handling

## Usage

### Basic Job Search and Application

```python
from find_jobs import FindJobsBot
from job_detail_to_form_transition_handler import JobDetailToFormTransitionHandler

# Initialize the bot
bot = FindJobsBot(headless=False, preferences=preferences)

# Run the automation
success = bot.run_bot("https://www.reed.co.uk/")
```

### Standalone Form Filling

```python
from fill_and_submit_job_form import ApplicationFiller

# Create application filler
app_filler = ApplicationFiller(page, preferences)

# Fill the application
success = app_filler.fill_application(
    on_success_callback=lambda: print("✅ Application completed"),
    on_failure_callback=lambda: print("❌ Application failed")
)
```

### Custom Navigation Handler

```python
from job_detail_to_form_transition_handler import JobDetailToFormTransitionHandler

# Create transition handler
handler = JobDetailToFormTransitionHandler(page)

# Navigate to form
result = handler.navigate_to_form(preferences)
if result.success and result.form_ready:
    print("✅ Form is ready for filling")
```

## Advanced Features

### Iframe Detection and Handling
The system automatically detects when forms are inside iframes and switches context accordingly:

- **Automatic Detection**: Scans for visible iframes containing form elements
- **Context Switching**: Seamlessly switches between main page and iframe contexts
- **Field Interaction**: All form operations work within the correct context
- **Screenshot Support**: Takes screenshots of both main page and iframe content

### Smart Field Detection
Uses Gemini 2.5 Pro to intelligently identify form fields:

- **Field Type Recognition**: Automatically detects input types (text, select, radio, checkbox, file)
- **Context Awareness**: Understands field relationships and form structure
- **Preference Mapping**: Maps user preferences to appropriate form fields
- **Validation**: Ensures fields are visible and interactive before interaction

### Job Evaluation System
Intelligent job matching using AI:

- **Preference Analysis**: Evaluates job fit based on user criteria
- **Scoring System**: Provides numerical scores and recommendations
- **Keyword Matching**: Identifies matching skills and requirements
- **Filtering**: Automatically filters out unsuitable positions

## Error Handling and Recovery

### Robust Error Management
- **API Error Handling**: Retries with exponential backoff for API failures
- **Element Interaction**: Multiple fallback strategies for element interaction
- **Page State Recovery**: Detects and recovers from unexpected page states
- **User Intervention**: Pauses automation when manual input is required

### Fallback Mechanisms
- **Selector Validation**: Multiple validation strategies for CSS selectors
- **Alternative Approaches**: Different methods for element interaction
- **Screenshot Debugging**: Comprehensive debugging with highlighted screenshots
- **Graceful Degradation**: Continues operation even when some features fail

## Testing

### Run Tests
```bash
# Test the application filler
python test_application_filler_robust.py

# Test the transition handler
python test_transition_handler.py
```

### Debug Mode
- Set `headless=False` to see the automation in action
- Use `page.pause()` for manual intervention
- Take screenshots with `take_screenshot()` method
- Enable highlighted element screenshots for debugging

## Architecture

### State Machine Design
The system uses a state machine approach to handle different page types:

1. **Search Page** → Fill search form → Submit
2. **Results Page** → Find job listings → Evaluate jobs
3. **Job Detail Page** → Extract job info → Evaluate fit
4. **Application Page** → Fill form → Submit application

### Modular Design
- **Separation of Concerns**: Each component handles specific functionality
- **Extensible Architecture**: Easy to add new job sites and form types
- **Reusable Components**: Components can be used independently
- **Clean Interfaces**: Well-defined interfaces between components

## Limitations and Considerations

### Current Limitations
- **API Dependencies**: Requires Google AI API access and quota
- **Browser Compatibility**: Primarily tested with Chrome/Chromium
- **Form Complexity**: May struggle with highly dynamic or complex forms
- **Rate Limiting**: API calls are subject to rate limits

### Best Practices
- **Test Thoroughly**: Always test on target job sites before production use
- **Monitor API Usage**: Track API calls to avoid quota issues
- **User Supervision**: Monitor automation for unexpected behavior
- **Regular Updates**: Keep dependencies updated for compatibility

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

For issues and questions:
1. Check existing issues in the repository
2. Create a new issue with detailed information
3. Include screenshots and error messages when possible
4. Provide steps to reproduce the issue

---

**Note**: This automation engine is designed for educational and personal use. Always comply with job site terms of service and respect rate limits and usage policies.
