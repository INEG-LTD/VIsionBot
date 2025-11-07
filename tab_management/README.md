# Tab Management - Phase 1: Foundation

## Overview

Phase 1 implements the foundational tab management system for multi-tab agent workflows. This provides the infrastructure for tracking, switching, and managing browser tabs.

## Components

### TabInfo
Metadata dataclass that stores information about a browser tab:
- Tab ID, URL, title
- Purpose/description
- Agent ownership
- Creation and access timestamps
- Completion status
- Custom metadata

### TabManager
Core manager class that handles:
- Tab registration and tracking
- Tab switching
- Tab closing with automatic fallback
- New tab detection
- Tab filtering and querying
- Orphaned tab cleanup

## Usage

### Basic Usage

```python
from tab_management import TabManager
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    context = browser.new_context()
    
    # Create TabManager
    tab_manager = TabManager(context)
    
    # Register a tab
    page1 = context.new_page()
    page1.goto("https://example.com")
    tab1_id = tab_manager.register_tab(
        page=page1,
        purpose="main",
        agent_id="agent_1"
    )
    
    # Create another tab
    page2 = context.new_page()
    page2.goto("https://google.com")
    tab2_id = tab_manager.register_tab(
        page=page2,
        purpose="search",
        agent_id="agent_1"
    )
    
    # Switch between tabs
    tab_manager.switch_to_tab(tab1_id)
    assert tab_manager.active_tab_id == tab1_id
    
    # Close a tab
    tab_manager.close_tab(tab1_id, switch_to=tab2_id)
    
    # List tabs
    all_tabs = tab_manager.list_tabs()
    agent_tabs = tab_manager.list_tabs(agent_id="agent_1")
    
    browser.close()
```

### Integration with BrowserVisionBot

TabManager is automatically initialized when creating a BrowserVisionBot:

```python
from vision_bot import BrowserVisionBot
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch()
    context = browser.new_context()
    page = context.new_page()
    
    bot = BrowserVisionBot(page=page)
    bot.start()
    
    # TabManager is available
    assert bot.tab_manager is not None
    
    # Initial page is registered
    assert len(bot.tab_manager.tabs) == 1
    
    # New tabs are automatically detected
    new_page = context.new_page()
    bot.switch_to_page(new_page)
    assert len(bot.tab_manager.tabs) == 2
```

## API Reference

### TabManager Methods

- `register_tab(page, purpose, agent_id=None, metadata=None)` → `str`: Register a new tab
- `get_tab_info(tab_id)` → `Optional[TabInfo]`: Get tab metadata
- `list_tabs(agent_id=None)` → `List[TabInfo]`: List all tabs (optionally filtered)
- `get_active_tab()` → `Optional[TabInfo]`: Get currently active tab
- `switch_to_tab(tab_id)` → `bool`: Switch to a different tab
- `close_tab(tab_id, switch_to=None)` → `bool`: Close a tab
- `detect_new_tab(page)` → `Optional[str]`: Detect and register a new tab
- `mark_tab_completed(tab_id)` → `bool`: Mark tab as completed
- `update_tab_info(tab_id, **kwargs)` → `bool`: Update tab information
- `get_tabs_by_purpose(purpose)` → `List[TabInfo]`: Get tabs by purpose
- `cleanup_orphaned_tabs()` → `int`: Clean up closed tabs

## Testing

Run Phase 1 tests:

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

Or use pytest directly:

```bash
# All tests
pytest tests/unit/test_tab_info.py tests/unit/test_tab_manager.py tests/integration/

# Specific test file
pytest tests/unit/test_tab_info.py -v

# With coverage
pytest tests/unit/ --cov=tab_management --cov-report=html
```

## Test Coverage

### Unit Tests
- ✅ TabInfo creation and validation
- ✅ TabInfo methods (update_access, mark_completed, etc.)
- ✅ TabManager registration
- ✅ TabManager retrieval and listing
- ✅ Tab switching
- ✅ Tab closing
- ✅ New tab detection
- ✅ Tab info updates
- ✅ Tab filtering by purpose/agent

### Integration Tests
- ✅ Real browser tab management
- ✅ Tab switching with real pages
- ✅ Tab closing with automatic switching
- ✅ New tab detection
- ✅ Tab filtering
- ✅ Integration with BrowserVisionBot

## Next Steps

Phase 1 provides the foundation for:
- **Phase 2**: Tab decision making (LLM-based decisions)
- **Phase 3**: Sub-agent infrastructure
- **Phase 4**: Sub-agent execution
- **Phase 5**: Full integration

