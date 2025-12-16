"""
Integration test for agent decision-making: select vs click for inputs with lists.

This test verifies that when an agent encounters an input field with a list of
suggestions below it, it should use `click:` on the option, NOT `select:`.

The select handler has been updated to:
1. Detect early if an element is NOT actually a select (not a <select> tag, no role="combobox"/"listbox")
2. Fail immediately with helpful error message suggesting to use 'click:' instead
3. Provide guidance to the agent when select methods don't work

This test validates:
- If agent incorrectly uses 'select:', the select handler will reject it early
- Agent should then use 'click:' on the suggestion item
- Clicking the suggestion fills the input field
- Form validation passes when input is correctly filled

If this test fails, it indicates the agent needs better guidance about when to use
click vs select, OR the select handler needs additional improvements.
"""
import time
from pathlib import Path
import tempfile
import os

import pytest
import time
import tempfile
import os

from vision_bot import BrowserVisionBot
from bot_config import BotConfig, ModelConfig, ExecutionConfig, ElementConfig, DebugConfig, BrowserConfig
from ai_utils import ReasoningLevel


# HTML fixture with a form that has an input field showing suggestions
TEST_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Location Form Test</title>
    <style>
        body { font-family: Arial; padding: 40px; max-width: 600px; margin: 0 auto; }
        .form-group { margin: 20px 0; }
        label { display: block; margin-bottom: 8px; font-weight: bold; }
        input[type="text"] { 
            width: 100%; 
            padding: 10px; 
            font-size: 16px;
            border: 2px solid #ccc;
            border-radius: 4px;
        }
        input[type="text"]:focus {
            border-color: #4CAF50;
            outline: none;
        }
        .suggestions-list { 
            border: 1px solid #ccc; 
            background: white;
            max-height: 200px;
            overflow-y: auto;
            margin-top: 5px;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            display: none;
        }
        .suggestions-list.visible {
            display: block;
        }
        .suggestion-item { 
            padding: 12px; 
            cursor: pointer;
            border-bottom: 1px solid #eee;
            transition: background 0.2s;
        }
        .suggestion-item:hover { 
            background: #f0f0f0; 
        }
        .suggestion-item:last-child {
            border-bottom: none;
        }
        button[type="submit"] {
            background: #4CAF50;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 4px;
            font-size: 16px;
            cursor: pointer;
            margin-top: 20px;
        }
        button[type="submit"]:hover {
            background: #45a049;
        }
        button[type="submit"]:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        .error-message {
            color: red;
            margin-top: 10px;
            display: none;
        }
        .error-message.show {
            display: block;
        }
    </style>
</head>
<body>
    <h1>Job Application Form</h1>
    <form id="application-form">
        <div class="form-group">
            <label for="location-input">Current Location <span style="color: red;">*</span></label>
            <input 
                type="text" 
                id="location-input" 
                name="location"
                placeholder="Start typing your location..."
                required
                autocomplete="off"
            />
            <div class="suggestions-list" id="location-suggestions">
                <div class="suggestion-item" id="location-0" data-value="Basle-Country, CHE">Basle-Country, CHE</div>
                <div class="suggestion-item" id="location-1" data-value="Kountri, GRC">Kountri, GRC</div>
                <div class="suggestion-item" id="location-2" data-value="Kountrí, Trifylia, GRC">Kountrí, Trifylia, GRC</div>
            </div>
            <div class="error-message" id="location-error">Please select a location from the suggestions.</div>
        </div>
        
        <button type="submit" id="submit-btn">Submit Application</button>
        <div id="form-status" style="margin-top: 20px; font-weight: bold;"></div>
    </form>
    
    <script>
        const locationInput = document.getElementById('location-input');
        const suggestionsList = document.getElementById('location-suggestions');
        const submitBtn = document.getElementById('submit-btn');
        const formStatus = document.getElementById('form-status');
        const locationError = document.getElementById('location-error');
        
        let suggestionsVisible = false;
        
        // Show suggestions when input is focused or has text
        locationInput.addEventListener('focus', function() {
            suggestionsList.classList.add('visible');
            suggestionsVisible = true;
        });
        
        locationInput.addEventListener('input', function() {
            if (this.value.trim().length > 0) {
                suggestionsList.classList.add('visible');
                suggestionsVisible = true;
            } else {
                suggestionsList.classList.remove('visible');
                suggestionsVisible = false;
            }
        });
        
        // Hide suggestions when clicking outside
        document.addEventListener('click', function(e) {
            if (!locationInput.contains(e.target) && !suggestionsList.contains(e.target)) {
                suggestionsList.classList.remove('visible');
                suggestionsVisible = false;
            }
        });
        
        // Handle suggestion clicks
        suggestionsList.querySelectorAll('.suggestion-item').forEach(item => {
            item.addEventListener('click', function() {
                locationInput.value = this.dataset.value;
                suggestionsList.classList.remove('visible');
                suggestionsVisible = false;
                locationError.classList.remove('show');
                // Trigger input event to validate
                locationInput.dispatchEvent(new Event('input', { bubbles: true }));
            });
        });
        
        // Form validation
        const form = document.getElementById('application-form');
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            locationError.classList.remove('show');
            
            // Check if location is filled with a valid suggestion value
            const locationValue = locationInput.value.trim();
            const validLocations = ['Basle-Country, CHE', 'Kountri, GRC', 'Kountrí, Trifylia, GRC'];
            
            if (!locationValue) {
                formStatus.textContent = '❌ Form validation failed: Location field is empty!';
                formStatus.style.color = 'red';
                locationError.classList.add('show');
                return false;
            }
            
            if (!validLocations.includes(locationValue)) {
                formStatus.textContent = '❌ Form validation failed: Invalid location value. Did you use select instead of click?';
                formStatus.style.color = 'red';
                locationError.classList.add('show');
                return false;
            }
            
            // Success
            formStatus.textContent = '✅ Form submitted successfully! Location: ' + locationValue;
            formStatus.style.color = 'green';
            submitBtn.disabled = true;
            return false;
        });
        
        // Also validate on input change
        locationInput.addEventListener('blur', function() {
            const locationValue = this.value.trim();
            const validLocations = ['Basle-Country, CHE', 'Kountri, GRC', 'Kountrí, Trifylia, GRC'];
            if (locationValue && !validLocations.includes(locationValue)) {
                locationError.classList.add('show');
            } else {
                locationError.classList.remove('show');
            }
        });
    </script>
</body>
</html>
"""


@pytest.fixture(scope="function")
def temp_html_file():
    """Create a temporary HTML file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
        f.write(TEST_HTML)
        temp_path = f.name
    yield temp_path
    # Cleanup
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture(scope="function")
def bot():
    """Create a BrowserVisionBot instance for testing."""
    from browser_provider import create_browser_provider
    
    config = BotConfig(
        model=ModelConfig(
            agent_model="gpt-4o-mini",  # Use a faster model for tests
            command_model="gpt-4o-mini",
            reasoning_level=ReasoningLevel.NONE
        ),
        execution=ExecutionConfig(
            max_attempts=10
        ),
        elements=ElementConfig(
            overlay_mode="all",
            include_textless_overlays=True,
            overlay_only_planning=True
        ),
        logging=DebugConfig(
            debug_mode=False  # Less verbose for tests
        ),
        browser=BrowserConfig(
            provider_type="local",
            headless=True,
            apply_stealth=False
        )
    )
    browser_provider = create_browser_provider(config.browser)
    bot = BrowserVisionBot(config=config, browser_provider=browser_provider)
    bot.start()
    yield bot
    try:
        bot.end()
    except Exception:
        pass


def test_agent_should_click_not_select_for_input_with_list(bot, temp_html_file):
    """
    Test that when an agent encounters an input field with suggestions,
    it should use `click:` on the suggestion, NOT `select:`.
    
    This is verified by:
    1. Agent types in the location field
    2. Suggestions appear
    3. Agent should click on a suggestion (which fills the input)
    4. Form submission succeeds
    
    If agent incorrectly uses `select:`, the input remains empty and form fails.
    """
    url = f"file://{temp_html_file}"
    bot.page.goto(url)
    
    # Give the page time to load
    time.sleep(1)
    
    # Execute task: fill the location field and submit
    result = bot.execute_task(
        "Type 'Bas' in the location field, then click on 'Basle-Country, CHE' from the suggestions list, then click Submit Application button.",
        max_iterations=10,
        base_knowledge=[
            "When an input field shows a list of suggestions below it, those are clickable items - use 'click:' command on them, NOT 'select:'",
            "The 'select:' command only works for <select> dropdown elements, not for input fields with suggestion lists",
            "If you use 'select:' on a regular input field, the input will remain empty and form validation will fail",
            "After clicking a suggestion, the input field should be filled with that value",
            "Then click the Submit Application button to submit the form"
        ]
    )
    
    # Wait a bit for any form submission to process
    time.sleep(0.5)
    
    # Check if the form was successfully submitted
    status_text = bot.page.text_content("#form-status") or ""
    
    # Verify the location input has a value (proving click worked, not select)
    location_value = bot.page.eval_on_selector("#location-input", "el => el.value") or ""
    
    print(f"\n=== Test Results ===")
    print(f"Location value: '{location_value}'")
    print(f"Form status: '{status_text}'")
    print(f"Result success: {result.success}")
    print(f"Result reasoning: {result.reasoning}")
    
    # The key assertion: location must be filled correctly
    # If agent used 'select:' instead of 'click:', the input would be empty or wrong
    assert location_value == "Basle-Country, CHE", \
        f"Location field should be filled with 'Basle-Country, CHE'. Got: '{location_value}'. " \
        f"This indicates the agent may have used 'select:' instead of 'click:'. " \
        f"Form status: '{status_text}'. Result: {result.reasoning}"
    
    # Also verify form validation would pass (or already passed)
    valid_locations = ['Basle-Country, CHE', 'Kountri, GRC', 'Kountrí, Trifylia, GRC']
    assert location_value in valid_locations, \
        f"Location value '{location_value}' is not in valid suggestions. " \
        f"This suggests the input wasn't filled by clicking a suggestion."


def test_agent_with_explicit_instruction_to_use_click(bot, temp_html_file):
    """
    Test with explicit instruction to use click, to verify the agent follows it.
    """
    url = f"file://{temp_html_file}"
    bot.page.goto(url)
    time.sleep(1)
    
    result = bot.execute_task(
        "Type 'Bas' in the location field to see suggestions, then CLICK on 'Basle-Country, CHE' suggestion (do NOT use select command). Then click Submit Application.",
        max_iterations=10,
        base_knowledge=[
            "When you see suggestions appear below an input field, use 'click:' command on the suggestion, NOT 'select:'",
            "The 'select:' command only works for HTML <select> elements",
            "Input fields with suggestion lists are NOT select elements - you must click the suggestion item",
            "After clicking a suggestion, the input field will be filled with that value",
            "Then click the Submit Application button"
        ]
    )
    
    time.sleep(0.5)
    location_value = bot.page.eval_on_selector("#location-input", "el => el.value") or ""
    status_text = bot.page.text_content("#form-status") or ""
    
    print(f"\n=== Test Results ===")
    print(f"Location value: '{location_value}'")
    print(f"Form status: '{status_text}'")
    print(f"Result success: {result.success}")
    
    # Verify location was filled correctly by clicking (not selecting)
    assert location_value == "Basle-Country, CHE", \
        f"Location should be 'Basle-Country, CHE' but got '{location_value}'. " \
        f"Agent may have used 'select:' instead of 'click:'. Result: {result.reasoning}"

