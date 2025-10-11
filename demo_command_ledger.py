"""
Demo: Using the command ledger to track command execution with IDs.
"""
from vision_bot import BrowserVisionBot
from action_executor import PostActionContext
from models import ActionType


def track_command_callback(ctx: PostActionContext):
    """Example callback that uses command ID information"""
    if ctx.command_id:
        # Get the command record from the ledger
        bot = ctx.page # We would need access to bot, but for demo purposes...
        print(f"\n📋 Action executed from command: {ctx.command_id}")
        
        if ctx.command_lineage and len(ctx.command_lineage) > 1:
            print(f"   └─ Command hierarchy: {' → '.join(ctx.command_lineage)}")


if __name__ == "__main__":
    # Initialize bot
    bot = BrowserVisionBot(save_gif=False)
    bot.start()
    
    # Register callback to track commands
    bot.action_executor.register_post_action_callback(track_command_callback)
    
    # Use custom command IDs
    bot.goto("https://www.reed.co.uk/")
    
    # Example 1: Manual command ID
    bot.act("click: the search button", command_id="search-btn-click")
    
    # Example 2: Auto-generated ID
    bot.act("type: 'python developer' into: the search field")
    
    # Example 3: Register prompts with ID
    bot.register_prompts([
        "click: job listing",
        "click: apply button"
    ], "job-application-flow", command_id="job-app-ref")
    
    # Access the command ledger
    print("\n📊 Command Ledger Stats:")
    stats = bot.command_ledger.get_stats()
    print(f"   Total commands: {stats['total_commands']}")
    print(f"   By status: {stats['by_status']}")
    print(f"   Total duration: {stats['total_duration']:.2f}s")
    print(f"   Average duration: {stats['average_duration']:.3f}s")
    
    # Query commands
    print("\n🔍 Completed Commands:")
    from command_ledger import CommandStatus
    completed = bot.command_ledger.filter_records(status=CommandStatus.COMPLETED)
    for record in completed:
        print(f"   - {record.id}: {record.command[:50]}... ({record.duration:.2f}s)")
    
    # Get execution tree
    print("\n🌳 Execution Tree:")
    import json
    tree = bot.command_ledger.get_execution_tree()
    print(json.dumps(tree, indent=2))
    
    bot.end()

