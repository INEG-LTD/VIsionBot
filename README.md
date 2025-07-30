# Job Application Automation Engine

A comprehensive Python-based automation engine for job applications using Playwright and GPT-4.1 for intelligent form filling.

## Features

- **Intelligent Form Detection**: Uses GPT-4.1 to analyze and identify form fields
- **Smart Dropdown Handling**: GPT-4.1-powered dropdown field interaction with HTML snapshots
- **Multi-Page Support**: Handles complex multi-step application processes
- **Iframe Support**: Automatically detects and handles forms inside iframes
- **Smart Field Mapping**: Maps user preferences to appropriate form fields
- **Error Recovery**: Robust error handling and fallback mechanisms
- **Page Type Detection**: Automatically detects different page types (search, results, application, etc.)

## Iframe Detection and Handling

The application filler now includes comprehensive iframe detection and handling capabilities:

### How It Works

1. **Iframe Detection**: The system automatically scans the page for visible iframes
2. **Form Element Detection**: Checks each iframe for form elements (inputs, selects, textareas)
3. **Context Switching**: If forms are found in iframes, the system switches to iframe context
4. **Field Interaction**: All form interactions (typing, clicking, selecting) work within the iframe context
5. **Submit Button Detection**: Searches for submit buttons in both main page and iframe contexts

### Key Methods

- `detect_and_handle_iframes()`: Detects iframes and determines if form fields are inside them
- `find_form_fields_in_iframe()`: Finds form fields within a specific iframe using GPT-4.1 analysis
- `fill_all_form_inputs()`: Automatically handles both main page and iframe contexts
- `find_and_click_submit_button()`: Searches for submit buttons in both contexts

### Benefits

- **Automatic Detection**: No manual configuration needed
- **Seamless Operation**: Works transparently whether forms are in main page or iframes
- **Robust Fallbacks**: Falls back to traditional selectors if GPT analysis fails
- **Error Handling**: Graceful handling of iframe access issues

## GPT-4.1 Dropdown Handling

The application filler now uses GPT-4.1 for intelligent dropdown field handling:

### How It Works

1. **HTML Snapshot**: Takes a snapshot of the current page HTML
2. **Dropdown Detection**: Uses GPT-4.1 to find the dropdown field from the HTML
3. **Field Interaction**: Clicks the dropdown field to open options
4. **Options Snapshot**: Takes another HTML snapshot to see available options
5. **Option Selection**: Uses GPT-4.1 to find and click the correct option

### Key Methods

- `_handle_custom_dropdown()`: Handles dropdowns in main page context using GPT-4.1
- `_handle_custom_dropdown_iframe()`: Handles dropdowns in iframe context using GPT-4.1
- `_find_dropdown_field_with_gpt()`: Uses GPT-4.1 to find dropdown fields from HTML
- `_find_and_click_option_with_gpt()`: Uses GPT-4.1 to find and click options

### Benefits

- **Intelligent Recognition**: GPT-4.1 understands complex dropdown structures
- **Universal Compatibility**: Works with traditional selects, custom dropdowns, React components, etc.
- **Context Awareness**: Understands the relationship between dropdown fields and their options
- **Robust Matching**: Handles variations in option text and dropdown structure

### Example Usage

```python
from application_filler import ApplicationFiller

# Initialize with preferences
preferences = {
    "name": "John Doe",
    "email": "john@example.com",
    "phone": "+1-555-123-4567",
    # ... other preferences
}

# Create application filler
app_filler = ApplicationFiller(page, preferences)

# The system automatically detects and handles iframes
success = app_filler.fill_application()
```

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up your OpenAI API key
4. Configure your preferences

## Testing

Run the iframe detection test:

```bash
python test_iframe_detection.py
```

This will create a test page with an iframe containing form elements and demonstrate the detection and filling capabilities.

## Configuration

Set your preferences in the configuration file or pass them directly to the ApplicationFiller constructor.

## License

MIT License 