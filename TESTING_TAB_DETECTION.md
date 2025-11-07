# Testing Tab Detection

## Quick Start

### Option 1: Run the Interactive Test Script

```bash
python test_tab_detection.py
```

This provides a menu with 3 test options:
1. **Basic tab detection** - Tests TabManager with BrowserVisionBot
2. **Automatic detection from click** - Tests detecting tabs opened from link clicks
3. **TabManager direct testing** - Tests TabManager without the bot

### Option 2: Run Unit/Integration Tests

```bash
# Run all Phase 1 tests
python tests/run_phase1_tests.py --all

# Run unit tests only
python tests/run_phase1_tests.py --unit

# Run integration tests only
python tests/run_phase1_tests.py --integration

# With verbose output
python tests/run_phase1_tests.py --all --verbose
```

### Option 3: Use pytest directly

```bash
# All tests
pytest tests/unit/test_tab_info.py tests/unit/test_tab_manager.py tests/integration/ -v

# Specific test file
pytest tests/integration/test_tab_switching.py -v

# Specific test
pytest tests/integration/test_tab_switching.py::TestTabSwitching::test_detect_new_tab -v
```

## Manual Testing Examples

### Example 1: Basic Tab Detection

```python
from playwright.sync_api import sync_playwright
from vision_bot import BrowserVisionBot

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()
    
    bot = BrowserVisionBot(page=page)
    bot.start()
    
    # Check TabManager is initialized
    print(f"TabManager available: {bot.tab_manager is not None}")
    print(f"Initial tabs: {len(bot.tab_manager.list_tabs())}")
    
    # Create a new tab
    new_page = context.new_page()
    new_page.goto("https://google.com")
    
    # Switch to it (should auto-detect)
    bot.switch_to_page(new_page)
    
    # Check tabs
    print(f"Tabs after new tab: {len(bot.tab_manager.list_tabs())}")
    for tab in bot.tab_manager.list_tabs():
        print(f"  - {tab.tab_id}: {tab.url}")
    
    bot.close()
    browser.close()
```

### Example 2: Test Automatic Detection from Click

```python
from playwright.sync_api import sync_playwright
from vision_bot import BrowserVisionBot
import time

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    context = browser.new_context()
    page = context.new_page()
    
    bot = BrowserVisionBot(page=page)
    bot.start()
    
    # Navigate to a page with a link that opens in new tab
    page.goto("https://example.com")
    
    # Inject a link that opens in new tab
    page.evaluate("""
        const link = document.createElement('a');
        link.href = 'https://google.com';
        link.target = '_blank';
        link.textContent = 'Open Google';
        link.id = 'test-link';
        document.body.appendChild(link);
    """)
    
    tabs_before = len(bot.tab_manager.list_tabs())
    print(f"Tabs before click: {tabs_before}")
    
    # Click the link
    page.click("#test-link")
    time.sleep(2)  # Wait for new tab
    
    tabs_after = len(bot.tab_manager.list_tabs())
    print(f"Tabs after click: {tabs_after}")
    
    if tabs_after > tabs_before:
        print("✅ New tab detected!")
        for tab in bot.tab_manager.list_tabs():
            if tab.purpose == "auto_detected":
                print(f"  New tab: {tab.url}")
    
    bot.close()
    browser.close()
```

### Example 3: Test Tab Switching

```python
from tab_management import TabManager
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    context = browser.new_context()
    
    tab_manager = TabManager(context)
    
    # Create multiple tabs
    page1 = context.new_page()
    page1.goto("https://example.com")
    tab1 = tab_manager.register_tab(page1, "tab1", "agent_1")
    
    page2 = context.new_page()
    page2.goto("https://google.com")
    tab2 = tab_manager.register_tab(page2, "tab2", "agent_1")
    
    # Switch between tabs
    print(f"Active: {tab_manager.get_active_tab().tab_id}")
    tab_manager.switch_to_tab(tab2)
    print(f"Active: {tab_manager.get_active_tab().tab_id}")
    
    browser.close()
```

## What to Look For

### ✅ Success Indicators

1. **TabManager initialized**: `bot.tab_manager is not None`
2. **Initial tab registered**: First page is registered as "main"
3. **New tabs detected**: Tabs opened from clicks are auto-detected
4. **Tab switching works**: Can switch between tabs
5. **Tab closing works**: Can close tabs with automatic fallback

### ⚠️ Common Issues

1. **Tabs not detected**: 
   - Make sure `bot.start()` was called (attaches listener)
   - Wait a moment after clicking (tabs open asynchronously)
   - Check browser console for errors

2. **Listener not attached**:
   - Ensure `bot.start()` is called before creating tabs
   - Check for errors in `_attach_new_page_listener()`

3. **Tab already exists**:
   - `detect_new_tab()` returns existing tab_id if page was already registered
   - This is expected behavior

## Debugging

### Enable Verbose Logging

The code already includes print statements. To see more:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Check Tab State

```python
# List all tabs
for tab in bot.tab_manager.list_tabs():
    print(f"Tab: {tab.tab_id}")
    print(f"  URL: {tab.url}")
    print(f"  Purpose: {tab.purpose}")
    print(f"  Agent: {tab.agent_id}")
    print(f"  Completed: {tab.is_completed}")
    print(f"  Metadata: {tab.metadata}")
```

### Check Active Tab

```python
active = bot.tab_manager.get_active_tab()
if active:
    print(f"Active tab: {active.tab_id} - {active.url}")
else:
    print("No active tab")
```

## Running in CI/Headless Mode

For automated testing, use `headless=True`:

```python
browser = p.chromium.launch(headless=True)
```

All tests should work in headless mode, but some visual verification won't be possible.

