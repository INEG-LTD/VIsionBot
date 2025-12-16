#!/usr/bin/env python3
"""
Simple standalone test to verify agent can see and select options from dropdowns.
"""
import sys
from pathlib import Path

from vision_bot import BrowserVisionBot
from bot_config import BotConfig, ActFunctionConfig
from browser_provider import create_browser_provider

FIXTURE = Path(__file__).parent / "tests" / "integration" / "select_fixtures.html"

def main():
    # Configure bot
    config = BotConfig()
    config.act_function = ActFunctionConfig(
        enable_target_context_guard=False,  # Simpler for testing
        enable_modifier=False,
    )
    config.logging.debug_mode = True
    
    browser_provider = create_browser_provider(config.browser)
    
    print("\n" + "="*60)
    print("AGENT SELECT FIELD INTEGRATION TEST")
    print("="*60 + "\n")
    
    with BrowserVisionBot(config=config, browser_provider=browser_provider) as bot:
        # Load fixture
        bot.page.goto(FIXTURE.as_uri())
        bot.page.evaluate("window.scrollTo(0, 0)")
        
        print("Test 1: Agent should see options and select 'Banana'")
        print("-" * 60)
        result = bot.execute_task(
            "Select 'Banana' from the favorite fruit dropdown",
            max_iterations=8
        )
        
        print(f"\nResult: {'✅ SUCCESS' if result.success else '❌ FAILED'}")
        print(f"Reasoning: {result.reasoning}")
        
        if result.success:
            value = bot.page.eval_on_selector("#basic-select", "el => el.value")
            text = bot.page.eval_on_selector("#basic-select", "el => el.options[el.selectedIndex].text")
            print(f"Selected value: {value}")
            print(f"Selected text: {text}")
            assert value == "banana", f"Expected 'banana', got '{value}'"
            print("✅ Verification passed!\n")
        else:
            print("❌ Agent failed to select\n")
            return 1
        
        # Test 2: Auto-select (no specific option)
        print("\nTest 2: Agent should auto-select (skip placeholder)")
        print("-" * 60)
        bot.page.goto(FIXTURE.as_uri())
        bot.page.evaluate("window.scrollTo(0, 0)")
        
        result = bot.execute_task(
            "Select any option from the beverage dropdown",
            max_iterations=8
        )
        
        print(f"\nResult: {'✅ SUCCESS' if result.success else '❌ FAILED'}")
        print(f"Reasoning: {result.reasoning}")
        
        if result.success:
            value = bot.page.eval_on_selector("#placeholder-select", "el => el.value")
            text = bot.page.eval_on_selector("#placeholder-select", "el => el.options[el.selectedIndex].text")
            print(f"Auto-selected value: {value}")
            print(f"Auto-selected text: {text}")
            # Should NOT be empty (placeholder)
            assert value != "", f"Selected placeholder instead of real option"
            assert value in ("coffee", "tea", "juice"), f"Got unexpected value: '{value}'"
            print("✅ Verification passed!\n")
        else:
            print("❌ Agent failed to auto-select\n")
            return 1
        
        # Test 3: Custom dropdown
        print("\nTest 3: Agent should handle custom dropdown")
        print("-" * 60)
        bot.page.goto(FIXTURE.as_uri())
        bot.page.evaluate("window.scrollTo(0, 1200)")  # Scroll to custom dropdown
        
        result = bot.execute_task(
            "Select 'Green' from the color picker",
            max_iterations=8
        )
        
        print(f"\nResult: {'✅ SUCCESS' if result.success else '❌ FAILED'}")
        print(f"Reasoning: {result.reasoning}")
        
        if result.success:
            text = bot.page.text_content("#custom-dropdown-trigger").strip()
            print(f"Custom dropdown selected: {text}")
            assert text == "Green", f"Expected 'Green', got '{text}'"
            print("✅ Verification passed!\n")
        else:
            print("❌ Agent failed to select from custom dropdown\n")
            return 1
    
    print("\n" + "="*60)
    print("ALL AGENT TESTS PASSED! 🎉")
    print("="*60 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())




