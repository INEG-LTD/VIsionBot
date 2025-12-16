"""
Test that the agent can successfully handle select, upload, and date fields.
This uses bot.execute_task to test the full agent flow.
"""
import sys
from pathlib import Path
import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from vision_bot import BrowserVisionBot
from bot_config import BotConfig, ActFunctionConfig
from browser_provider import create_browser_provider


FIXTURE_DIR = Path(__file__).parent
SELECT_FIXTURE = FIXTURE_DIR / "select_fixtures.html"


@pytest.fixture(scope="module")
def bot_config():
    """Create bot config with context guard disabled for cleaner testing"""
    config = BotConfig()
    config.act_function = ActFunctionConfig(
        enable_target_context_guard=False,
        enable_modifier=False,
    )
    return config


@pytest.fixture
def bot(bot_config):
    """Create a bot instance for testing"""
    browser_provider = create_browser_provider(bot_config.browser)
    bot = BrowserVisionBot(config=bot_config, browser_provider=browser_provider)
    bot.start()
    yield bot
    try:
        bot.end()
    except Exception:
        pass


def test_agent_handles_native_select(bot):
    """Test that agent can select from a native dropdown"""
    bot.page.goto(SELECT_FIXTURE.as_uri())
    bot.page.evaluate("window.scrollTo(0, 0)")
    
    # Agent should be able to select an option from basic select
    result = bot.execute_task(
        "Select 'Banana' from the favorite fruit dropdown",
        max_iterations=5
    )
    
    assert result.success, f"Failed: {result.reasoning}"
    
    # Verify selection
    value = bot.page.eval_on_selector("#basic-select", "el => el.value")
    assert value == "banana", f"Expected 'banana', got '{value}'"
    print(f"  ✅ Agent selected: {value}")


def test_agent_handles_select_with_placeholder(bot):
    """Test that agent can handle select with placeholder option"""
    bot.page.goto(SELECT_FIXTURE.as_uri())
    bot.page.evaluate("window.scrollTo(0, 0)")
    
    # Agent should skip placeholder and select a real option
    result = bot.execute_task(
        "Select 'Coffee' from the beverage dropdown",
        max_iterations=5
    )
    
    assert result.success, f"Failed: {result.reasoning}"
    
    value = bot.page.eval_on_selector("#placeholder-select", "el => el.value")
    assert value == "coffee", f"Expected 'coffee', got '{value}'"
    print(f"  ✅ Agent selected: {value}")


def test_agent_handles_optgroup_select(bot):
    """Test that agent can select from grouped options"""
    bot.page.goto(SELECT_FIXTURE.as_uri())
    bot.page.evaluate("window.scrollTo(0, 0)")
    
    result = bot.execute_task(
        "Select 'Steak' from the meal dropdown",
        max_iterations=5
    )
    
    assert result.success, f"Failed: {result.reasoning}"
    
    value = bot.page.eval_on_selector("#optgroup-select", "el => el.value")
    assert value == "steak", f"Expected 'steak', got '{value}'"
    print(f"  ✅ Agent selected: {value}")


def test_agent_handles_custom_dropdown(bot):
    """Test that agent can handle custom dropdown with button trigger"""
    bot.page.goto(SELECT_FIXTURE.as_uri())
    bot.page.evaluate("window.scrollTo(0, 1200)")  # Scroll to custom dropdown
    
    result = bot.execute_task(
        "Select 'Purple' from the color picker",
        max_iterations=5
    )
    
    assert result.success, f"Failed: {result.reasoning}"
    
    text = bot.page.text_content("#custom-dropdown-trigger").strip()
    assert text == "Purple", f"Expected 'Purple', got '{text}'"
    print(f"  ✅ Agent selected custom option: {text}")


def test_agent_auto_selects_from_dropdown(bot):
    """Test that agent can auto-select (pick any option) when not specified"""
    bot.page.goto(SELECT_FIXTURE.as_uri())
    bot.page.evaluate("window.scrollTo(0, 0)")
    
    # Agent should select ANY option (skipping placeholder)
    result = bot.execute_task(
        "Select any beverage from the beverage dropdown",
        max_iterations=5
    )
    
    assert result.success, f"Failed: {result.reasoning}"
    
    value = bot.page.eval_on_selector("#placeholder-select", "el => el.value")
    # Should be one of the real options, not empty placeholder
    assert value in ("coffee", "tea", "juice"), f"Got placeholder or invalid: '{value}'"
    print(f"  ✅ Agent auto-selected: {value}")


def test_agent_handles_date_input(bot):
    """Test that agent can fill a date input"""
    date_html = """
    <!DOCTYPE html>
    <html>
    <head><title>Date Test</title></head>
    <body>
        <h1>Event Registration</h1>
        <form>
            <label for="event-date">Event Date</label>
            <input type="date" id="event-date" name="event-date" required>
            <button type="submit">Submit</button>
        </form>
    </body>
    </html>
    """
    bot.page.set_content(date_html)
    
    result = bot.execute_task(
        "Set the event date to December 25, 2024",
        max_iterations=5
    )
    
    assert result.success, f"Failed: {result.reasoning}"
    
    value = bot.page.eval_on_selector("#event-date", "el => el.value")
    assert value == "2024-12-25", f"Expected '2024-12-25', got '{value}'"
    print(f"  ✅ Agent set date: {value}")


def test_agent_handles_file_upload(bot):
    """Test that agent can handle file upload"""
    # Create test file
    test_file = FIXTURE_DIR / "test_agent_upload.txt"
    test_file.write_text("Agent test content")
    
    upload_html = f"""
    <!DOCTYPE html>
    <html>
    <head><title>Upload Test</title></head>
    <body>
        <h1>Document Upload</h1>
        <form>
            <label for="document">Upload your document</label>
            <input type="file" id="document" name="document" required>
            <div id="status"></div>
        </form>
        <script>
            document.getElementById('document').addEventListener('change', (e) => {{
                const fileName = e.target.files[0]?.name || 'No file';
                document.getElementById('status').textContent = 'Uploaded: ' + fileName;
            }});
        </script>
    </body>
    </html>
    """
    bot.page.set_content(upload_html)
    
    try:
        result = bot.execute_task(
            f"Upload the file located at {test_file}",
            max_iterations=5
        )
        
        assert result.success, f"Failed: {result.reasoning}"
        
        # Verify file was uploaded
        status = bot.page.text_content("#status")
        assert "test_agent_upload.txt" in status, f"File not uploaded: {status}"
        print(f"  ✅ Agent uploaded: {status}")
    finally:
        test_file.unlink()


if __name__ == "__main__":
    # Run tests individually for debugging
    import sys
    
    config = BotConfig()
    config.act_function = ActFunctionConfig(
        enable_target_context_guard=False,
        enable_modifier=False,
    )
    browser_provider = create_browser_provider(config.browser)
    
    with BrowserVisionBot(config=config, browser_provider=browser_provider) as bot:
        print("\n=== Testing Agent Field Handling ===\n")
        
        # Test 1: Native select
        print("Test 1: Native select dropdown")
        bot.page.goto(SELECT_FIXTURE.as_uri())
        result = bot.execute_task("Select 'Cherry' from the favorite fruit dropdown", max_iterations=5)
        print(f"Result: {result.success} - {result.reasoning}\n")
        
        # Test 2: Select with placeholder
        print("Test 2: Select with placeholder")
        bot.page.goto(SELECT_FIXTURE.as_uri())
        result = bot.execute_task("Select 'Tea' from the beverage dropdown", max_iterations=5)
        print(f"Result: {result.success} - {result.reasoning}\n")
        
        # Test 3: Custom dropdown
        print("Test 3: Custom dropdown")
        bot.page.goto(SELECT_FIXTURE.as_uri())
        bot.page.evaluate("window.scrollTo(0, 1200)")
        result = bot.execute_task("Select 'Red' from the color picker", max_iterations=5)
        print(f"Result: {result.success} - {result.reasoning}\n")
        
        print("=== All agent tests completed ===")




