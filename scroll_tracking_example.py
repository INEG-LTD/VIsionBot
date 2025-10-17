"""
Example demonstrating scroll tracking in PostActionContext

The PostActionContext now includes:
- scroll_occurred: bool - Whether a scroll happened
- scroll_reason: Optional[ScrollReason] - Why the scroll occurred (enum value)

Possible ScrollReason enum values:
- ScrollReason.USER_ACTION: User explicitly requested a scroll action
- ScrollReason.DUPLICATE_REJECTION: Scrolled because duplicate elements were detected multiple times
- ScrollReason.DOM_UNCHANGED: Scrolled because DOM signature hasn't changed (likely due to deduplication)
- ScrollReason.EXPLORE_CONTENT: Scrolled to explore more content after successful plan execution
- ScrollReason.MANUAL: Manual scroll (default for programmatic scrolls)
"""

from action_executor import PostActionContext, ScrollReason
from vision_bot import BrowserVisionBot

bot = BrowserVisionBot()
bot.start()
bot.goto("https://example.com")

def post_action_callback(ctx: PostActionContext):
    """Example callback showing how to use scroll tracking"""
    
    # Check if a scroll occurred
    if ctx.scroll_occurred:
        print(f"🔄 Scroll detected! Reason: {ctx.scroll_reason}")
        
        # React to specific scroll reasons using enum values
        if ctx.scroll_reason == ScrollReason.DUPLICATE_REJECTION:
            print("   ⚠️ Page was scrolled due to duplicate content detection")
            print("   💡 You might want to wait or check for new content")
        
        elif ctx.scroll_reason == ScrollReason.DOM_UNCHANGED:
            print("   ⚠️ Page was scrolled because DOM hasn't changed")
            print("   💡 Goal might be stuck, consider alternative approach")
        
        elif ctx.scroll_reason == ScrollReason.USER_ACTION:
            print("   ✅ User requested scroll action")
        
        elif ctx.scroll_reason == ScrollReason.EXPLORE_CONTENT:
            print("   🔍 Scrolled to explore more content")
    
    # You can also check action details
    print(f"Action: {ctx.action_type}, Success: {ctx.success}")
    if ctx.command_id:
        print(f"Command ID: {ctx.command_id}")

# Register the callback
bot.action_executor.register_post_action_callback(post_action_callback)

# Now perform some actions
bot.act("click: learn more link")
bot.act("scroll: down")

bot.end()

