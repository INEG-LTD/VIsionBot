"""
Demo: Command Ledger Advanced Features
- Persistence (save/load)
- Comparison (compare runs)
- Logger Integration
"""
from vision_bot import BrowserVisionBot
from command_ledger import CommandLedger
import time


def demo_persistence():
    """Demo: Save and load ledger to/from files"""
    print("\n" + "="*80)
    print("DEMO 1: PERSISTENCE")
    print("="*80)
    
    bot = BrowserVisionBot(save_gif=False)
    bot.start()
    
    # Run some commands
    bot.goto("https://www.reed.co.uk/")
    bot.act("click: the search button", command_id="search-1")
    bot.act("type: 'python developer' into: search field", command_id="type-1")
    
    # Save ledger to file
    session_file = "ledgers/session_run1.json"
    bot.command_ledger.save_to_file(session_file)
    
    # Export human-readable summary
    bot.command_ledger.export_summary("ledgers/session_run1_summary.txt")
    
    print("\n✅ Saved ledger and summary!")
    
    # Load it back (simulate analyzing a previous run)
    print("\n📂 Loading ledger from file...")
    loaded_ledger = CommandLedger()
    loaded_ledger.load_from_file(session_file)
    
    print(f"✅ Loaded {len(loaded_ledger)} commands")
    stats = loaded_ledger.get_stats()
    print(f"   Total duration: {stats['total_duration']:.2f}s")
    print(f"   Status breakdown: {stats['by_status']}")
    
    bot.end()


def demo_comparison():
    """Demo: Compare two automation runs"""
    print("\n" + "="*80)
    print("DEMO 2: COMPARISON")
    print("="*80)
    
    # Simulate Run 1
    print("\n🏃 Run 1: Initial automation run...")
    bot1 = BrowserVisionBot(save_gif=False)
    bot1.start()
    bot1.goto("https://www.reed.co.uk/")
    bot1.act("click: the search button", command_id="search")
    bot1.act("type: 'python' into: search field", command_id="type-python")
    time.sleep(0.5)
    bot1.command_ledger.save_to_file("ledgers/run1.json")
    bot1.end()
    
    # Simulate Run 2 (modified workflow)
    print("\n🏃 Run 2: Modified automation run...")
    bot2 = BrowserVisionBot(save_gif=False)
    bot2.start()
    bot2.goto("https://www.reed.co.uk/")
    bot2.act("click: the search button", command_id="search")
    bot2.act("type: 'javascript' into: search field", command_id="type-js")  # Different
    bot2.act("click: filters button", command_id="filters")  # New command
    time.sleep(1.0)  # Simulate slower execution
    bot2.command_ledger.save_to_file("ledgers/run2.json")
    bot2.end()
    
    # Compare the two runs
    print("\n🔍 Comparing runs...")
    comparison = CommandLedger.load_and_compare("ledgers/run1.json", "ledgers/run2.json")
    comparison.print_summary()


def demo_logger_integration():
    """Demo: Integrate command ledger with bot logger"""
    print("\n" + "="*80)
    print("DEMO 3: LOGGER INTEGRATION")
    print("="*80)
    
    bot = BrowserVisionBot(save_gif=False)
    bot.start()
    
    # Enable logger integration
    bot.command_ledger.enable_logger_integration(bot.logger)
    
    print("\n✅ Logger integration enabled - all command events will be logged\n")
    
    # Run commands - they'll automatically be logged
    bot.goto("https://www.reed.co.uk/")
    bot.act("click: the search button", command_id="logged-search")
    bot.act("type: 'python' into: field", command_id="logged-type")
    
    print("\n📝 Check bot logs - command events should be included!")
    
    # View the session log
    try:
        bot.write_session_log()
        print("✅ Session log written with command tracking")
    except Exception as e:
        print(f"Note: Could not write session log: {e}")
    
    bot.end()


def demo_all_features():
    """Run all feature demos"""
    print("\n" + "="*80)
    print("COMMAND LEDGER - ADVANCED FEATURES DEMO")
    print("="*80)
    
    # Demo 1: Persistence
    demo_persistence()
    
    # Demo 2: Comparison
    demo_comparison()
    
    # Demo 3: Logger Integration
    demo_logger_integration()
    
    print("\n" + "="*80)
    print("ALL DEMOS COMPLETED")
    print("="*80)
    print("\n📁 Check the 'ledgers/' directory for saved files")
    print("📄 Check session logs for integrated command tracking")


if __name__ == "__main__":
    # Uncomment to run individual demos:
    # demo_persistence()
    # demo_comparison()
    # demo_logger_integration()
    
    # Or run all demos:
    demo_all_features()

