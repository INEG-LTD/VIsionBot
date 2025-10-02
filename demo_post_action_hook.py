"""
Simple demo: Post-action hooks that run after every action.
This example shows how to wait after each action or run custom logic.
"""
from vision_bot import BrowserVisionBot
from action_executor import PostActionContext
from models import ActionType


def wait_after_action(ctx: PostActionContext):
    """Wait 500ms after every successful action for stability"""
    if ctx.success:
        import time
        time.sleep(0.5)
        print(f"  ⏱️  Waited 500ms after {ctx.action_type.value}")


def scroll_after_click(ctx: PostActionContext):
    """Automatically scroll down 100px after every click"""
    if ctx.success and ctx.action_type == ActionType.CLICK:
        try:
            ctx.page.evaluate("window.scrollBy(0, 100)")
            print(f"  📜 Auto-scrolled 100px after click")
        except Exception as e:
            print(f"  ⚠️  Auto-scroll failed: {e}")


def press_tab_after_type(ctx: PostActionContext):
    """Press Tab after typing to move to the next field"""
    if ctx.success and ctx.action_type == ActionType.TYPE:
        try:
            ctx.page.keyboard.press("Tab")
            print(f"  ⇥ Pressed Tab to move to next field")
        except Exception as e:
            print(f"  ⚠️  Tab press failed: {e}")


if __name__ == "__main__":
    # Initialize bot
    bot = BrowserVisionBot(save_gif=False)
    bot.start()
    
    # Register post-action callbacks
    bot.action_executor.register_post_action_callback(wait_after_action)
    bot.action_executor.register_post_action_callback(scroll_after_click)
    bot.action_executor.register_post_action_callback(press_tab_after_type)
    
    # Now run your automation - callbacks will trigger after each action
    bot.goto("https://www.reed.co.uk/")
    bot.act("click: the search button")
    bot.act("type: 'python developer' into: the search field")
    
    print("\n✅ All actions completed with post-action hooks!")
    
    # Cleanup
    bot.end()

