"""
Integration test for automatic conversion from select to click.

This test verifies that when the select handler determines an element is not a select,
it automatically converts the action to a click action.
"""
import time
import tempfile
import os

import pytest
from playwright.sync_api import sync_playwright

from action_executor import ActionExecutor
from handlers.select_handler import SelectHandler, ShouldUseClickInsteadError
from models import ActionStep, PageElements, PageInfo, DetectedElement
from session_tracker import SessionTracker


def _normalized_box(box, doc_width, doc_height):
    """Convert Playwright bounding box to normalized Gemini-style box."""
    x_min = box["x"]
    y_min = box["y"]
    x_max = x_min + box["width"]
    y_max = y_min + box["height"]
    return [
        int(y_min / doc_height * 1000),
        int(x_min / doc_width * 1000),
        int(y_max / doc_height * 1000),
        int(x_max / doc_width * 1000),
    ]


def _page_info(page):
    viewport = page.viewport_size or {"width": 1280, "height": 720}
    doc = page.evaluate(
        """() => ({
            width: document.documentElement.clientWidth,
            height: document.documentElement.clientHeight,
            scrollX: window.scrollX,
            scrollY: window.scrollY,
            dpr: window.devicePixelRatio || 1,
            docWidth: document.documentElement.scrollWidth,
            docHeight: document.documentElement.scrollHeight,
        })"""
    )
    return PageInfo(
        width=viewport["width"],
        height=viewport["height"],
        scroll_x=doc["scrollX"],
        scroll_y=doc["scrollY"],
        url=page.url,
        title=page.title(),
        dpr=doc["dpr"],
        ss_pixel_w=viewport["width"],
        ss_pixel_h=viewport["height"],
        css_scale=1.0,
        doc_width=doc["docWidth"],
        doc_height=doc["docHeight"],
    )


def _detected_element(page, selector, overlay_number, description):
    """Create a DetectedElement from a selector."""
    box = page.locator(selector).bounding_box()
    assert box, f"Bounding box not found for selector {selector}"
    doc = page.evaluate(
        """() => ({
            docWidth: document.documentElement.scrollWidth,
            docHeight: document.documentElement.scrollHeight,
        })"""
    )
    box_2d = _normalized_box(box, doc["docWidth"], doc["docHeight"])
    return DetectedElement(
        element_label=selector,
        description=description,
        element_type="select",  # Simulate agent thinking it's a select
        is_clickable=True,
        box_2d=box_2d,
        section="content",
        field_subtype="select",
        confidence=0.9,
        requires_special_handling=True,
        overlay_number=overlay_number,
    )


# HTML fixture
TEST_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Auto Convert Test</title>
    <style>
        body { font-family: Arial; padding: 40px; }
        .input-with-list { margin: 20px 0; }
        .suggestions-list { 
            border: 1px solid #ccc; 
            background: white;
            max-height: 200px;
            margin-top: 5px;
        }
        .suggestion-item { 
            padding: 12px; 
            cursor: pointer;
            border-bottom: 1px solid #eee;
        }
        .suggestion-item:hover { background: #f0f0f0; }
        input { padding: 10px; width: 300px; }
    </style>
</head>
<body>
    <h1>Location Input Test</h1>
    
    <div class="input-with-list">
        <label>Location:</label>
        <input type="text" id="location-input" placeholder="Enter location" />
        <div class="suggestions-list" id="location-suggestions">
            <div class="suggestion-item" id="location-0" data-value="Basle-Country, CHE">Basle-Country, CHE</div>
            <div class="suggestion-item" id="location-1" data-value="Kountri, GRC">Kountri, GRC</div>
        </div>
    </div>
    
    <div id="result" style="margin-top: 20px; font-weight: bold;"></div>
    
    <script>
        const locationInput = document.getElementById('location-input');
        const suggestionsList = document.getElementById('location-suggestions');
        const resultDiv = document.getElementById('result');
        
        // Show suggestions when input is focused
        locationInput.addEventListener('focus', function() {
            suggestionsList.style.display = 'block';
        });
        
        // Handle suggestion clicks
        suggestionsList.querySelectorAll('.suggestion-item').forEach(item => {
            item.addEventListener('click', function() {
                locationInput.value = this.dataset.value;
                suggestionsList.style.display = 'none';
                resultDiv.textContent = '✅ Location selected: ' + this.dataset.value;
                resultDiv.style.color = 'green';
            });
        });
        
        // Hide suggestions when clicking outside
        document.addEventListener('click', function(e) {
            if (!locationInput.contains(e.target) && !suggestionsList.contains(e.target)) {
                suggestionsList.style.display = 'none';
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
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture(scope="function")
def page():
    """Create a test page."""
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        yield page
        browser.close()


def test_auto_convert_select_to_click(page, temp_html_file):
    """
    Test that when select handler fails, it automatically converts to click.
    """
    page.goto(f"file://{temp_html_file}")
    time.sleep(0.5)
    
    # Create executor with session tracker
    session_tracker = SessionTracker(page)
    executor = ActionExecutor(page, session_tracker)
    
    # Create element as if agent detected it as a select
    elem = _detected_element(page, "#location-0", 1, "location suggestion")
    
    # Create a select step (simulating agent choosing select)
    select_step = ActionStep(
        action="handle_select",
        overlay_index=1,
        select_option_text="Basle-Country, CHE"
    )
    
    # Set coordinates
    bbox = page.locator("#location-0").bounding_box()
    if bbox:
        select_step.x = int(bbox["x"] + bbox["width"] / 2)
        select_step.y = int(bbox["y"] + bbox["height"] / 2)
    
    # Create a plan with the step
    from models import VisionPlan
    plan = VisionPlan(
        action_steps=[select_step],
        detected_elements=PageElements(elements=[elem]),
        reasoning="Test plan",
        confidence=0.9
    )
    
    page_info = _page_info(page)
    
    # Execute the plan - should auto-convert select to click
    success = executor.execute_plan(plan, page_info)
    
    # Verify the location input was filled (proving click worked)
    location_value = page.eval_on_selector("#location-input", "el => el.value") or ""
    result_text = page.text_content("#result") or ""
    
    print(f"\n=== Test Results ===")
    print(f"Location value: '{location_value}'")
    print(f"Result text: '{result_text}'")
    print(f"Plan execution success: {success}")
    
    # The key assertion: location should be filled, proving auto-conversion worked
    assert location_value == "Basle-Country, CHE" or "Basle-Country" in result_text, \
        f"Location should be filled with 'Basle-Country, CHE' but got '{location_value}'. " \
        f"Auto-conversion from select to click may not have worked. Result: {result_text}"

