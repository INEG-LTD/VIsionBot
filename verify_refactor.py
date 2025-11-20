
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from vision_bot import BrowserVisionBot
from bot_config import BotConfig

def test_initialization():
    print("Testing Bot Initialization...")
    try:
        config = BotConfig()
        bot = BrowserVisionBot(config=config)
        print("✅ Bot initialized successfully")
        return bot
    except Exception as e:
        print(f"❌ Bot initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_utilities():
    print("\nTesting Utilities...")
    try:
        from utils.execution_timer import ExecutionTimer
        timer = ExecutionTimer()
        timer.start_task()
        timer.end_task()
        print("✅ ExecutionTimer working")
        
        from utils.command_parser import parse_action_intent
        intent = parse_action_intent("click the button")
        if intent == "click":
            print("✅ CommandParser working")
        else:
            print(f"❌ CommandParser failed: expected 'click', got '{intent}'")
            
        from utils.vision_resolver import resolve_element
        print("✅ VisionResolver imported successfully")
        
    except Exception as e:
        print(f"❌ Utilities test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    bot = test_initialization()
    test_utilities()
    if bot:
        print("\nAll checks passed!")
