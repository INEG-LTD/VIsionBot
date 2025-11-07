# How New Tab Detection Works

## Current Mechanism

New tab detection uses **Playwright's BrowserContext event system**. Here's how it works:

### 1. Event Listener Setup

When `BrowserVisionBot.start()` is called, it attaches a listener to the browser context:

```python
def _attach_new_page_listener(self) -> None:
    ctx = self.page.context
    
    def _on_new_page(new_page: Page) -> None:
        # This fires whenever a new page/tab is created
        # - Clicking a link with target="_blank"
        # - JavaScript window.open()
        # - Programmatic context.new_page()
        
        if self.tab_manager:
            tab_id = self.tab_manager.detect_new_tab(new_page)
        
        self.switch_to_page(new_page)
    
    ctx.on("page", _on_new_page)  # Playwright event
```

**Key Point**: Playwright's `"page"` event fires automatically when:
- A new tab is opened from a click (e.g., `<a target="_blank">`)
- JavaScript opens a new window/tab (`window.open()`)
- A new page is created programmatically (`context.new_page()`)

### 2. Tab Detection Logic

`TabManager.detect_new_tab()` uses Python's `id()` to track unique pages:

```python
def detect_new_tab(self, page: Page) -> Optional[str]:
    page_id = id(page)  # Unique object ID
    
    # Check if we've seen this page before
    if page_id in self._known_pages:
        # Already registered, return existing tab_id
        return existing_tab_id
    
    # New page - register it
    tab_id = self.register_tab(
        page=page,
        purpose="auto_detected",
        metadata={"auto_detected": True}
    )
    
    return tab_id
```

### 3. Detection Flow

```
User clicks link with target="_blank"
    ↓
Playwright detects new page
    ↓
BrowserContext fires "page" event
    ↓
_on_new_page() callback executes
    ↓
TabManager.detect_new_tab() checks if page is new
    ↓
If new: Register tab with "auto_detected" purpose
    ↓
Bot switches to new tab automatically
```

## Detection Scenarios

### ✅ Automatically Detected

1. **Link clicks with `target="_blank"`**
   ```html
   <a href="https://example.com" target="_blank">Open in new tab</a>
   ```
   - Playwright fires `"page"` event immediately
   - Tab is detected and registered

2. **JavaScript `window.open()`**
   ```javascript
   window.open("https://example.com", "_blank");
   ```
   - Playwright detects the new page
   - Event fires automatically

3. **Programmatic tab creation**
   ```python
   new_page = context.new_page()
   ```
   - If not registered manually, `detect_new_tab()` will catch it when accessed

### ⚠️ Edge Cases

1. **Tabs opened before listener is attached**
   - If tabs exist before `start()` is called, they won't be auto-detected
   - Solution: Manually register them or call `detect_new_tab()` when switching

2. **Tabs opened in different contexts**
   - Each `BrowserContext` has its own event system
   - Tabs in other contexts won't be detected
   - This is expected behavior (one TabManager per context)

3. **Tabs that close immediately**
   - If a tab closes before detection, it may not be registered
   - Solution: `cleanup_orphaned_tabs()` will handle it

## Alternative Detection Methods

### Method 1: Proactive Detection After Clicks

We could check for new tabs after every click:

```python
def _execute_click(self, ...):
    # Before click
    pages_before = set(context.pages)
    
    # Perform click
    element.click()
    
    # After click - check for new pages
    pages_after = set(context.pages)
    new_pages = pages_after - pages_before
    
    if new_pages:
        for new_page in new_pages:
            tab_manager.detect_new_tab(new_page)
```

**Pros**: More explicit, can track which click opened which tab  
**Cons**: More overhead, might miss tabs opened asynchronously

### Method 2: Popup Event (Specific Use Case)

For specific popup scenarios:

```python
with page.expect_popup() as popup_info:
    page.click("a[target='_blank']")
new_page = popup_info.value
```

**Pros**: Explicit, can wait for specific popup  
**Cons**: Only works for popups, not all new tabs

### Method 3: Periodic Page Enumeration

Check `context.pages` periodically:

```python
def check_for_new_tabs(self):
    current_pages = set(context.pages)
    known_page_ids = {id(tab.page) for tab in self.tabs.values()}
    
    for page in current_pages:
        if id(page) not in known_page_ids:
            self.detect_new_tab(page)
```

**Pros**: Catches tabs that might be missed  
**Cons**: Polling overhead, less efficient

## Current Implementation: Event-Based (Recommended)

The current **event-based approach** is the best because:

1. ✅ **Automatic**: No manual checking needed
2. ✅ **Efficient**: Only fires when tabs actually open
3. ✅ **Reliable**: Playwright handles all edge cases
4. ✅ **Immediate**: Detects tabs as soon as they're created

## Improving Detection

If you want to track **which action opened which tab**, we could enhance it:

```python
class TabManager:
    def __init__(self, ...):
        self._pending_tab_source = None  # Track what action might open a tab
    
    def mark_potential_tab_creation(self, action: str, element_info: dict):
        """Mark that the next tab might be from this action"""
        self._pending_tab_source = {
            "action": action,
            "element": element_info,
            "timestamp": time.time()
        }
    
    def detect_new_tab(self, page: Page) -> Optional[str]:
        # ... existing detection ...
        
        # If we have pending source info, add it to metadata
        if self._pending_tab_source:
            metadata["source_action"] = self._pending_tab_source["action"]
            metadata["source_element"] = self._pending_tab_source["element"]
            self._pending_tab_source = None
```

This would let us know: "This tab was opened by clicking the 'LinkedIn Profile' link"

## Summary

**Current Detection**: ✅ Event-based via Playwright's `"page"` event  
**When it fires**: Automatically when any new page/tab is created  
**Reliability**: High - Playwright handles all browser-specific behavior  
**Performance**: Efficient - no polling or manual checks needed  

The system works well as-is, but can be enhanced to track tab sources if needed for Phase 2 (Tab Decision Making).

