# Scroll Tracking in PostActionContext

The `PostActionContext` now includes comprehensive scroll tracking to help you understand when and why scrolls occur during automation.

## Fields

### `scroll_occurred: bool`
Indicates whether a scroll event happened.

### `scroll_reason: Optional[ScrollReason]`
An enum value explaining why the scroll occurred.

## ScrollReason Enum

```python
from action_executor import ScrollReason

class ScrollReason(Enum):
    USER_ACTION = "user_action"           # User explicitly requested a scroll action
    DUPLICATE_REJECTION = "duplicate_rejection"  # Scrolled due to repeated duplicate element detection
    DOM_UNCHANGED = "dom_unchanged"       # Scrolled because DOM signature hasn't changed
    EXPLORE_CONTENT = "explore_content"   # Scrolled to explore more content after successful plan execution
    MANUAL = "manual"                     # Default for programmatic scrolls
```

## Usage Example

```python
from action_executor import PostActionContext, ScrollReason
from vision_bot import BrowserVisionBot

bot = BrowserVisionBot()
bot.start()

def post_action_callback(ctx: PostActionContext):
    # Check if a scroll occurred
    if ctx.scroll_occurred:
        print(f"🔄 Scroll detected! Reason: {ctx.scroll_reason}")
        
        # React to specific scroll reasons
        if ctx.scroll_reason == ScrollReason.DUPLICATE_REJECTION:
            print("   ⚠️ Page scrolled due to duplicate content")
            # Maybe wait for new content to load
            
        elif ctx.scroll_reason == ScrollReason.DOM_UNCHANGED:
            print("   ⚠️ Page scrolled because DOM hasn't changed")
            # Goal might be stuck
            
        elif ctx.scroll_reason == ScrollReason.USER_ACTION:
            print("   ✅ User requested scroll")
            
        elif ctx.scroll_reason == ScrollReason.EXPLORE_CONTENT:
            print("   🔍 Scrolled to explore more content")

# Register the callback
bot.action_executor.register_post_action_callback(post_action_callback)

# Perform actions - scrolls will be tracked automatically
bot.act("scroll: down")  # This will have reason=ScrollReason.USER_ACTION
```

## When Scrolls Occur

### USER_ACTION
- User explicitly calls `bot.act("scroll: down")` or similar
- Any scroll action in the command queue

### DUPLICATE_REJECTION
- The focus manager detects duplicate elements multiple times
- Triggered when `duplicate_rejection_count >= duplicate_rejection_threshold`
- Location: `focus_manager.py` line ~388

### DOM_UNCHANGED
- The DOM signature hasn't changed since the last attempt
- Indicates the page isn't responding to interactions as expected
- Usually due to deduplication blocking all available elements
- Location: `vision_bot.py` line ~853

### EXPLORE_CONTENT
- Plan executed successfully but no goals were achieved
- Bot scrolls to find new content instead of retrying the same view
- Location: `vision_bot.py` line ~945

### MANUAL
- Default reason for programmatic scrolls
- Used when calling `page_utils.scroll_page()` without specifying a reason

## Accessing Scroll Reason Values

You can access the string value of the enum:

```python
if ctx.scroll_occurred:
    print(f"Scroll reason: {ctx.scroll_reason.value}")  # Prints "duplicate_rejection", "user_action", etc.
```

Or compare directly with enum values:

```python
if ctx.scroll_reason == ScrollReason.DUPLICATE_REJECTION:
    # Handle duplicate rejection scroll
    pass
```

## Benefits

1. **Better Debugging**: Know exactly why your automation is scrolling
2. **Conditional Logic**: React differently based on scroll reasons
3. **Performance Monitoring**: Track when duplicate content is causing issues
4. **Type Safety**: Enum provides autocomplete and type checking

