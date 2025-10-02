"""
Example: Using post-action callbacks to run predetermined actions after each interaction.

This demonstrates how to register callbacks that execute after every click, type, scroll, etc.
"""
from action_executor import PostActionContext
from models import ActionType


def example_wait_callback(ctx: PostActionContext):
    """Wait 500ms after every successful action"""
    if ctx.success:
        import time
        time.sleep(0.5)
        print(f"  🕐 Post-action wait: 500ms after {ctx.action_type.value}")


def example_scroll_after_click(ctx: PostActionContext):
    """Scroll down 50px after every successful click"""
    if ctx.success and ctx.action_type == ActionType.CLICK:
        try:
            ctx.page.evaluate("window.scrollBy(0, 50)")
            print(f"  📜 Post-action scroll: 50px down after click at {ctx.coordinates}")
        except Exception as e:
            print(f"  ⚠️ Post-action scroll failed: {e}")


def example_screenshot_callback(ctx: PostActionContext):
    """Take a screenshot after every action (successful or not)"""
    try:
        filename = f"post_action_{ctx.action_type.value}_{int(ctx.step.overlay_index or 0)}.png"
        ctx.page.screenshot(path=filename)
        print(f"  📸 Saved screenshot: {filename}")
    except Exception as e:
        print(f"  ⚠️ Screenshot failed: {e}")


def example_conditional_callback(ctx: PostActionContext):
    """Run different actions based on the action type and success"""
    if not ctx.success:
        print(f"  ❌ Action {ctx.action_type.value} failed: {ctx.error_message}")
        # Could retry or log error
        return
    
    if ctx.action_type == ActionType.TYPE:
        # After typing, press Tab to move to next field
        print(f"  ⌨️ Typed '{ctx.step.text_to_type}', pressing Tab to move to next field")
        try:
            ctx.page.keyboard.press("Tab")
        except Exception as e:
            print(f"  ⚠️ Tab press failed: {e}")
    
    elif ctx.action_type == ActionType.CLICK:
        # After clicking, wait for potential page navigation or modal
        import time
        time.sleep(0.3)
        print(f"  🖱️ Clicked, waiting for potential navigation...")
    
    elif ctx.action_type == ActionType.SCROLL:
        # After scrolling, wait for lazy-loaded content
        import time
        time.sleep(0.5)
        print(f"  📜 Scrolled to {ctx.coordinates}, waiting for content to load...")


def example_logging_callback(ctx: PostActionContext):
    """Log detailed information about every action"""
    status = "✅" if ctx.success else "❌"
    print(f"\n{status} Post-Action Log:")
    print(f"  Action Type: {ctx.action_type.value}")
    print(f"  Success: {ctx.success}")
    print(f"  Page URL: {ctx.page_info.url}")
    print(f"  Page Size: {ctx.page_info.width}x{ctx.page_info.height}")
    print(f"  Scroll Position: ({ctx.page_info.scroll_x}, {ctx.page_info.scroll_y})")
    
    if ctx.coordinates:
        print(f"  Coordinates: {ctx.coordinates}")
    
    if ctx.step.overlay_index is not None:
        print(f"  Target Element: Overlay #{ctx.step.overlay_index}")
    
    if ctx.step.text_to_type:
        print(f"  Text Typed: {ctx.step.text_to_type}")
    
    if ctx.error_message:
        print(f"  Error: {ctx.error_message}")
    
    print(f"  Detected Elements: {len(ctx.elements.elements)}")
    print()


# ==============================================================================
# Usage Example with BrowserVisionBot
# ==============================================================================

if __name__ == "__main__":
    from vision_bot import BrowserVisionBot
    
    # Initialize bot
    bot = BrowserVisionBot(save_gif=True)
    bot.start()
    
    # Register callbacks - the action_executor is directly accessible
    
    # Example 1: Register a simple wait callback
    bot.action_executor.register_post_action_callback(example_wait_callback)
    
    # Example 2: Scroll after every click
    bot.action_executor.register_post_action_callback(example_scroll_after_click)
    
    # Example 3: Take screenshots after every action (commented out to avoid clutter)
    # bot.action_executor.register_post_action_callback(example_screenshot_callback)
    
    # Example 4: Conditional actions based on action type
    bot.action_executor.register_post_action_callback(example_conditional_callback)
    
    # Example 5: Detailed logging
    bot.action_executor.register_post_action_callback(example_logging_callback)
    
    # Now any actions will trigger ALL registered callbacks
    bot.goto("https://example.com")
    bot.act("click: the first link")
    bot.act("type: 'hello world' into: the search box")
    
    # You can unregister specific callbacks
    bot.action_executor.unregister_post_action_callback(example_logging_callback)
    
    # Or clear all callbacks at once
    # bot.action_executor.clear_post_action_callbacks()
    
    bot.end()

