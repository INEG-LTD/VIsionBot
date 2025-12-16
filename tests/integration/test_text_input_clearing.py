"""
Integration test to verify that text input fields are cleared before typing.
This ensures that when typing into a field with existing text, the old text
is removed before the new text is entered.
"""
import sys
import time
from pathlib import Path
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from action_executor import ActionExecutor
from models import ActionStep, ActionType, PageElements, PageInfo
from browser_provider import create_browser_provider
from bot_config import BotConfig
from utils.page_utils import PageUtils
from session_tracker import SessionTracker
from interaction_deduper import InteractionDeduper
from action_ledger import ActionLedger
from utils.event_logger import EventLogger
from unittest.mock import Mock


FIXTURE_DIR = Path(__file__).parent
TEXT_INPUT_FIXTURE = FIXTURE_DIR / "text_input_fixtures.html"


@pytest.fixture(scope="module")
def browser_context():
    """Create a browser context for testing"""
    from playwright.sync_api import sync_playwright
    playwright = sync_playwright().start()
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context(viewport={"width": 1280, "height": 800})
    yield context
    browser.close()
    playwright.stop()


@pytest.fixture
def page(browser_context):
    """Create a new page for each test"""
    page = browser_context.new_page()
    yield page
    page.close()


@pytest.fixture
def action_executor(page):
    """Create an ActionExecutor instance for testing"""
    # Set up event logger before creating executor (it will use get_event_logger)
    from utils.event_logger import set_event_logger
    event_logger = EventLogger(debug_mode=True)
    set_event_logger(event_logger)
    
    # Create required dependencies
    page_utils = PageUtils(page)
    session_tracker = SessionTracker(page)
    deduper = InteractionDeduper()
    action_ledger = ActionLedger()
    
    executor = ActionExecutor(
        page=page,
        session_tracker=session_tracker,
        page_utils=page_utils,
        deduper=deduper,
        action_ledger=action_ledger,
        gif_recorder=None,
    )
    
    return executor


def test_basic_text_input_clearing(page, action_executor):
    """Test that typing into a text input with existing text clears it first"""
    page.goto(TEXT_INPUT_FIXTURE.as_uri())
    time.sleep(0.5)  # Wait for page to load
    
    # Get the input field coordinates
    input_selector = "#basic-input"
    box = page.locator(input_selector).bounding_box()
    assert box is not None, "Input field not found"
    
    x = int(box['x'] + box['width'] / 2)
    y = int(box['y'] + box['height'] / 2)
    
    # Verify initial value
    initial_value = page.input_value(input_selector)
    assert initial_value == "Initial text value", f"Expected initial value, got: {initial_value}"
    print(f"  Initial value: '{initial_value}'")
    
    # Create a type action
    step = ActionStep(
        action=ActionType.TYPE,
        x=x,
        y=y,
        text_to_type="New text value"
    )
    
    # Create mock elements (not needed for this test but required by executor)
    elements = PageElements(elements=[])
    page_info = action_executor.page_utils.get_page_info()
    
    # Execute the type action
    success = action_executor._execute_type(step, elements, page_info)
    assert success, "Type action should succeed"
    
    # Wait a bit for the typing to complete
    time.sleep(0.5)
    
    # Verify the field contains only the new text (not the old text)
    final_value = page.input_value(input_selector)
    assert final_value == "New text value", f"Expected 'New text value', got: '{final_value}'"
    print(f"  Final value: '{final_value}'")
    assert "Initial text value" not in final_value, "Old text should not be present"


def test_multiple_typing_attempts(page, action_executor):
    """Test typing into the same field multiple times - each should clear previous text"""
    page.goto(TEXT_INPUT_FIXTURE.as_uri())
    time.sleep(0.5)
    
    input_selector = "#basic-input"
    box = page.locator(input_selector).bounding_box()
    assert box is not None
    
    x = int(box['x'] + box['width'] / 2)
    y = int(box['y'] + box['height'] / 2)
    
    # Type first value
    step1 = ActionStep(
        action=ActionType.TYPE,
        x=x,
        y=y,
        text_to_type="First attempt"
    )
    elements = PageElements(elements=[])
    page_info = action_executor.page_utils.get_page_info()
    
    success1 = action_executor._execute_type(step1, elements, page_info)
    assert success1
    time.sleep(0.5)
    
    value1 = page.input_value(input_selector)
    assert value1 == "First attempt", f"First typing failed: got '{value1}'"
    print(f"  After first type: '{value1}'")
    
    # Type second value (should clear first)
    step2 = ActionStep(
        action=ActionType.TYPE,
        x=x,
        y=y,
        text_to_type="Second attempt"
    )
    
    success2 = action_executor._execute_type(step2, elements, page_info)
    assert success2
    time.sleep(0.5)
    
    value2 = page.input_value(input_selector)
    assert value2 == "Second attempt", f"Second typing failed: got '{value2}'"
    assert "First attempt" not in value2, "First value should be cleared"
    print(f"  After second type: '{value2}'")
    
    # Type third value (should clear second)
    step3 = ActionStep(
        action=ActionType.TYPE,
        x=x,
        y=y,
        text_to_type="Third attempt"
    )
    
    success3 = action_executor._execute_type(step3, elements, page_info)
    assert success3
    time.sleep(0.5)
    
    value3 = page.input_value(input_selector)
    assert value3 == "Third attempt", f"Third typing failed: got '{value3}'"
    assert "Second attempt" not in value3, "Second value should be cleared"
    assert "First attempt" not in value3, "First value should still be cleared"
    print(f"  After third type: '{value3}'")


def test_email_input_clearing(page, action_executor):
    """Test that email input fields are also cleared before typing"""
    page.goto(TEXT_INPUT_FIXTURE.as_uri())
    time.sleep(0.5)
    
    input_selector = "#email-input"
    box = page.locator(input_selector).bounding_box()
    assert box is not None
    
    x = int(box['x'] + box['width'] / 2)
    y = int(box['y'] + box['height'] / 2)
    
    # Verify initial value
    initial_value = page.input_value(input_selector)
    assert initial_value == "old@example.com"
    print(f"  Initial email: '{initial_value}'")
    
    # Type new email
    step = ActionStep(
        action=ActionType.TYPE,
        x=x,
        y=y,
        text_to_type="new@test.com"
    )
    
    elements = PageElements(elements=[])
    page_info = action_executor.page_utils.get_page_info()
    
    success = action_executor._execute_type(step, elements, page_info)
    assert success
    time.sleep(0.5)
    
    final_value = page.input_value(input_selector)
    assert final_value == "new@test.com", f"Expected 'new@test.com', got: '{final_value}'"
    assert "old@example.com" not in final_value, "Old email should be cleared"
    print(f"  Final email: '{final_value}'")


def test_textarea_clearing(page, action_executor):
    """Test that textarea fields are cleared before typing"""
    page.goto(TEXT_INPUT_FIXTURE.as_uri())
    time.sleep(0.5)
    
    input_selector = "#textarea-input"
    box = page.locator(input_selector).bounding_box()
    assert box is not None
    
    x = int(box['x'] + box['width'] / 2)
    y = int(box['y'] + box['height'] / 2)
    
    # Verify initial value
    initial_value = page.input_value(input_selector)
    assert "initial text" in initial_value.lower()
    print(f"  Initial textarea value length: {len(initial_value)} chars")
    
    # Type new text
    step = ActionStep(
        action=ActionType.TYPE,
        x=x,
        y=y,
        text_to_type="New textarea content"
    )
    
    elements = PageElements(elements=[])
    page_info = action_executor.page_utils.get_page_info()
    
    success = action_executor._execute_type(step, elements, page_info)
    assert success
    time.sleep(0.5)
    
    final_value = page.input_value(input_selector)
    assert final_value == "New textarea content", f"Expected 'New textarea content', got: '{final_value}'"
    assert len(final_value) < len(initial_value), "New text should be shorter than initial text"
    print(f"  Final textarea value: '{final_value}'")


def test_empty_field_typing(page, action_executor):
    """Test that typing into an empty field still works correctly"""
    page.goto(TEXT_INPUT_FIXTURE.as_uri())
    time.sleep(0.5)
    
    input_selector = "#empty-input"
    # Scroll to ensure the field is visible
    page.locator(input_selector).scroll_into_view_if_needed()
    time.sleep(0.3)
    
    box = page.locator(input_selector).bounding_box()
    assert box is not None
    
    x = int(box['x'] + box['width'] / 2)
    y = int(box['y'] + box['height'] / 2)
    
    # Verify field is empty
    initial_value = page.input_value(input_selector)
    assert initial_value == "", f"Field should be empty, got: '{initial_value}'"
    
    # Type into empty field - disable refinement to ensure we hit the right element
    step = ActionStep(
        action=ActionType.TYPE,
        x=x,
        y=y,
        text_to_type="New content"
    )
    
    elements = PageElements(elements=[])
    page_info = action_executor.page_utils.get_page_info()
    
    # Execute with refinement disabled to avoid hitting wrong element
    success = action_executor._execute_type(step, elements, page_info, allow_refinement=False)
    assert success
    time.sleep(0.5)
    
    final_value = page.input_value(input_selector)
    assert final_value == "New content", f"Expected 'New content', got: '{final_value}'"
    print(f"  Typed into empty field: '{final_value}'")


if __name__ == "__main__":
    # Run tests individually for debugging
    from playwright.sync_api import sync_playwright
    
    playwright = sync_playwright().start()
    browser = playwright.chromium.launch(headless=False)  # Set to False to see browser
    context = browser.new_context(viewport={"width": 1280, "height": 800})
    page = context.new_page()
    
    try:
        from utils.event_logger import set_event_logger
        event_logger = EventLogger(debug_mode=True)
        set_event_logger(event_logger)
        
        page_utils = PageUtils(page)
        session_tracker = SessionTracker(page)
        deduper = InteractionDeduper()
        action_ledger = ActionLedger()
        
        executor = ActionExecutor(
            page=page,
            session_tracker=session_tracker,
            page_utils=page_utils,
            deduper=deduper,
            action_ledger=action_ledger,
            gif_recorder=None,
        )
        
        print("\n=== Testing Text Input Clearing ===\n")
        
        # Test 1: Basic clearing
        print("Test 1: Basic text input clearing")
        page.goto(TEXT_INPUT_FIXTURE.as_uri())
        time.sleep(1)
        
        input_selector = "#basic-input"
        box = page.locator(input_selector).bounding_box()
        x = int(box['x'] + box['width'] / 2)
        y = int(box['y'] + box['height'] / 2)
        
        initial = page.input_value(input_selector)
        print(f"  Initial: '{initial}'")
        
        step = ActionStep(action=ActionType.TYPE, x=x, y=y, text_to_type="New text")
        elements = PageElements(elements=[])
        page_info = executor.page_utils.get_page_info()
        
        executor._execute_type(step, elements, page_info)
        time.sleep(1)
        
        final = page.input_value(input_selector)
        print(f"  Final: '{final}'")
        print(f"  Success: {final == 'New text'}\n")
        
        # Test 2: Multiple attempts
        print("Test 2: Multiple typing attempts")
        for i, text in enumerate(["First", "Second", "Third"], 1):
            step = ActionStep(action=ActionType.TYPE, x=x, y=y, text_to_type=text)
            executor._execute_type(step, elements, page_info)
            time.sleep(0.5)
            value = page.input_value(input_selector)
            print(f"  Attempt {i}: '{value}'")
        
        print("\n=== All tests completed ===")
        
    finally:
        browser.close()
        playwright.stop()
