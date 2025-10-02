# Post-Action Hooks

Post-action hooks allow you to run custom actions **automatically after every browser interaction** (click, type, scroll, press, etc.).

## Overview

The callback system provides complete context about each action, including:
- Action type (click, type, scroll, etc.)
- Success status
- Page information (URL, dimensions, scroll position)
- Element coordinates
- Detected elements
- Error messages (if failed)
- Direct access to the Playwright `Page` object

## Quick Start

```python
from vision_bot import BrowserVisionBot
from action_executor import PostActionContext
from models import ActionType

# Define your callback
def my_callback(ctx: PostActionContext):
    if ctx.success and ctx.action_type == ActionType.CLICK:
        print(f"Clicked at {ctx.coordinates}")
        # Run any custom action here

# Initialize bot and register callback
bot = BrowserVisionBot()
bot.start()
bot.action_executor.register_post_action_callback(my_callback)

# Now all actions will trigger the callback
bot.goto("https://example.com")
bot.act("click: the first button")
```

## PostActionContext Fields

The callback receives a `PostActionContext` object with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `action_type` | `ActionType` | The type of action (CLICK, TYPE, SCROLL, PRESS, etc.) |
| `success` | `bool` | Whether the action succeeded |
| `step` | `ActionStep` | Complete action details (overlay_index, text_to_type, etc.) |
| `page_info` | `PageInfo` | Page state (URL, dimensions, scroll position) |
| `elements` | `PageElements` | Detected elements on the page |
| `coordinates` | `Optional[Tuple[int, int]]` | Action coordinates (if applicable) |
| `error_message` | `Optional[str]` | Error message if the action failed |
| `page` | `Page` | Playwright Page object for custom actions |

## Common Use Cases

### 1. Wait After Every Action

```python
def wait_after_action(ctx: PostActionContext):
    """Add a 500ms delay after every successful action"""
    if ctx.success:
        import time
        time.sleep(0.5)

bot.action_executor.register_post_action_callback(wait_after_action)
```

### 2. Scroll After Clicks

```python
def scroll_after_click(ctx: PostActionContext):
    """Scroll down after every click to reveal content"""
    if ctx.success and ctx.action_type == ActionType.CLICK:
        ctx.page.evaluate("window.scrollBy(0, 100)")

bot.action_executor.register_post_action_callback(scroll_after_click)
```

### 3. Press Tab After Typing

```python
def tab_after_type(ctx: PostActionContext):
    """Automatically move to next field after typing"""
    if ctx.success and ctx.action_type == ActionType.TYPE:
        ctx.page.keyboard.press("Tab")

bot.action_executor.register_post_action_callback(tab_after_type)
```

### 4. Take Screenshots on Failure

```python
def screenshot_on_failure(ctx: PostActionContext):
    """Save a screenshot whenever an action fails"""
    if not ctx.success:
        filename = f"error_{ctx.action_type.value}_{int(time.time())}.png"
        ctx.page.screenshot(path=filename)
        print(f"Saved error screenshot: {filename}")

bot.action_executor.register_post_action_callback(screenshot_on_failure)
```

### 5. Wait for Network Idle

```python
def wait_for_network(ctx: PostActionContext):
    """Wait for network to be idle after clicks"""
    if ctx.success and ctx.action_type == ActionType.CLICK:
        try:
            ctx.page.wait_for_load_state("networkidle", timeout=3000)
        except:
            pass  # Continue if timeout

bot.action_executor.register_post_action_callback(wait_for_network)
```

### 6. Conditional Actions

```python
def conditional_actions(ctx: PostActionContext):
    """Different actions based on action type"""
    if not ctx.success:
        return  # Skip on failure
    
    if ctx.action_type == ActionType.TYPE:
        ctx.page.keyboard.press("Tab")
    elif ctx.action_type == ActionType.CLICK:
        import time
        time.sleep(0.3)  # Wait for animations
    elif ctx.action_type == ActionType.SCROLL:
        import time
        time.sleep(0.5)  # Wait for lazy-load

bot.action_executor.register_post_action_callback(conditional_actions)
```

### 7. Logging and Analytics

```python
def log_actions(ctx: PostActionContext):
    """Log all actions to a file"""
    with open("action_log.txt", "a") as f:
        status = "SUCCESS" if ctx.success else "FAILED"
        f.write(f"{status}: {ctx.action_type.value} at {ctx.page_info.url}\n")

bot.action_executor.register_post_action_callback(log_actions)
```

## Managing Callbacks

### Register a Callback

```python
bot.action_executor.register_post_action_callback(my_callback)
```

### Unregister a Specific Callback

```python
bot.action_executor.unregister_post_action_callback(my_callback)
```

### Clear All Callbacks

```python
bot.action_executor.clear_post_action_callbacks()
```

### Multiple Callbacks

You can register multiple callbacks - they all execute in order:

```python
bot.action_executor.register_post_action_callback(wait_after_action)
bot.action_executor.register_post_action_callback(scroll_after_click)
bot.action_executor.register_post_action_callback(log_actions)
# All three will run after each action
```

## Error Handling

Callbacks are wrapped in try-catch blocks. If a callback throws an exception, it will be caught and logged, and other callbacks will still execute:

```python
def risky_callback(ctx: PostActionContext):
    raise Exception("This won't crash your bot")

bot.action_executor.register_post_action_callback(risky_callback)
# The bot continues running even if this callback fails
```

## Performance Considerations

- Callbacks run **synchronously** after each action
- Keep callbacks fast to avoid slowing down automation
- Avoid blocking operations in callbacks
- Use timeouts for network waits

## Advanced Example

```python
from vision_bot import BrowserVisionBot
from action_executor import PostActionContext
from models import ActionType
import time

class ActionTracker:
    def __init__(self):
        self.click_count = 0
        self.type_count = 0
        self.errors = []
    
    def track_action(self, ctx: PostActionContext):
        """Track statistics about actions"""
        if ctx.success:
            if ctx.action_type == ActionType.CLICK:
                self.click_count += 1
            elif ctx.action_type == ActionType.TYPE:
                self.type_count += 1
        else:
            self.errors.append({
                'type': ctx.action_type.value,
                'error': ctx.error_message,
                'url': ctx.page_info.url
            })
    
    def print_stats(self):
        print(f"\n📊 Action Statistics:")
        print(f"  Clicks: {self.click_count}")
        print(f"  Types: {self.type_count}")
        print(f"  Errors: {len(self.errors)}")

# Use the tracker
tracker = ActionTracker()
bot = BrowserVisionBot()
bot.start()
bot.action_executor.register_post_action_callback(tracker.track_action)

# Run your automation
bot.goto("https://example.com")
bot.act("click: button")
bot.act("type: 'hello' into: input")

# Print statistics
tracker.print_stats()
bot.end()
```

## See Also

- `example_post_action_callback.py` - Comprehensive examples
- `demo_post_action_hook.py` - Simple working demo
- `action_executor.py` - Implementation details

