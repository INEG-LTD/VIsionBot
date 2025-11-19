"""
Examples demonstrating the Browser Provider pattern.

This file shows how to use different browser providers with BrowserVisionBot
for flexibility, testing, and remote browser support.
"""

from bot_config import BotConfig
from browser_provider import (
    BrowserConfig,
    LocalPlaywrightProvider,
    RemoteBrowserProvider,
    PersistentContextProvider,
    create_browser_provider
)
from vision_bot import BrowserVisionBot


def example_1_default_provider():
    """Example 1: Default local browser provider (automatic)"""
    print("=" * 60)
    print("Example 1: Default Provider")
    print("=" * 60)
    
    # Bot automatically creates LocalPlaywrightProvider from config
    config = BotConfig()
    bot = BrowserVisionBot(config=config)
    
    print("✓ Bot created with default local browser provider")
    print(f"  - Provider type: {config.browser.provider_type}")
    print(f"  - Headless: {config.browser.headless}")
    print(f"  - Viewport: {config.browser.viewport_width}x{config.browser.viewport_height}")
    print()


def example_2_custom_browser_config():
    """Example 2: Custom browser configuration"""
    print("=" * 60)
    print("Example 2: Custom Browser Config")
    print("=" * 60)
    
    # Configure browser settings in BotConfig
    config = BotConfig(
        browser=BrowserConfig(
            headless=True,
            viewport_width=1920,
            viewport_height=1080,
            apply_stealth=True
        )
    )
    
    bot = BrowserVisionBot(config=config)
    
    print("✓ Bot created with custom browser configuration")
    print(f"  - Headless: {config.browser.headless}")
    print(f"  - Viewport: {config.browser.viewport_width}x{config.browser.viewport_height}")
    print(f"  - Stealth: {config.browser.apply_stealth}")
    print()


def example_3_dependency_injection():
    """Example 3: Dependency injection with custom provider"""
    print("=" * 60)
    print("Example 3: Dependency Injection")
    print("=" * 60)
    
    # Create provider manually
    browser_config = BrowserConfig(
        headless=False,
        viewport_width=1280,
        viewport_height=800
    )
    provider = LocalPlaywrightProvider(browser_config)
    
    # Inject provider into bot
    bot = BrowserVisionBot(
        config=BotConfig(),
        browser_provider=provider
    )
    
    print("✓ Bot created with injected browser provider")
    print(f"  - Provider: LocalPlaywrightProvider")
    print(f"  - Headless: {browser_config.headless}")
    print()


def example_4_remote_browser():
    """Example 4: Remote browser (Browserbase/Browserless)"""
    print("=" * 60)
    print("Example 4: Remote Browser")
    print("=" * 60)
    
    # Configure for remote browser
    config = BotConfig(
        browser=BrowserConfig(
            provider_type="remote",
            remote_cdp_url="wss://connect.browserbase.com?apiKey=YOUR_API_KEY"
        )
    )
    
    # Note: This will fail without a valid CDP URL
    # bot = BrowserVisionBot(config=config)
    # bot.start()
    
    print("✓ Configuration for remote browser")
    print(f"  - Provider type: {config.browser.provider_type}")
    print(f"  - CDP URL: {config.browser.remote_cdp_url}")
    print("  - (Not starting - requires valid API key)")
    print()


def example_5_persistent_context():
    """Example 5: Persistent browser context (reuse profile)"""
    print("=" * 60)
    print("Example 5: Persistent Context")
    print("=" * 60)
    
    # Use existing Chrome profile
    config = BotConfig(
        browser=BrowserConfig(
            provider_type="persistent",
            user_data_dir="/Users/yourname/Library/Application Support/Google/Chrome/Default"
        )
    )
    
    # Note: Update path to your actual Chrome profile
    print("✓ Configuration for persistent context")
    print(f"  - Provider type: {config.browser.provider_type}")
    print(f"  - User data dir: {config.browser.user_data_dir}")
    print("  - (Not starting - requires valid profile path)")
    print()


def example_6_shared_browser():
    """Example 6: Share browser across multiple bots"""
    print("=" * 60)
    print("Example 6: Shared Browser")
    print("=" * 60)
    
    # Create one provider
    provider = LocalPlaywrightProvider(
        BrowserConfig(headless=True)
    )
    
    # Use same provider for multiple bots
    bot1 = BrowserVisionBot(
        config=BotConfig(),
        browser_provider=provider
    )
    
    bot2 = BrowserVisionBot(
        config=BotConfig(),
        browser_provider=provider
    )
    
    print("✓ Two bots sharing same browser provider")
    print("  - Bot 1 and Bot 2 will use same browser instance")
    print("  - Each gets its own page/tab")
    print("  - Reduces memory footprint")
    print()


def example_7_factory_pattern():
    """Example 7: Using factory function"""
    print("=" * 60)
    print("Example 7: Factory Pattern")
    print("=" * 60)
    
    # Use factory to create provider from config
    browser_config = BrowserConfig(
        provider_type="local",
        headless=True,
        viewport_width=1600
    )
    
    provider = create_browser_provider(browser_config)
    
    bot = BrowserVisionBot(
        config=BotConfig(),
        browser_provider=provider
    )
    
    print("✓ Bot created using factory pattern")
    print(f"  - Provider created from config")
    print(f"  - Type: {type(provider).__name__}")
    print()


def example_8_testing_setup():
    """Example 8: Setup for testing (mock browser)"""
    print("=" * 60)
    print("Example 8: Testing Setup")
    print("=" * 60)
    
    # In tests, you would use MockBrowserProvider
    # from unittest.mock import Mock
    # mock_page = Mock(spec=Page)
    # provider = MockBrowserProvider(BrowserConfig(), mock_page=mock_page)
    # bot = BrowserVisionBot(config=BotConfig(), browser_provider=provider)
    
    print("✓ Testing setup pattern")
    print("  - Use MockBrowserProvider in tests")
    print("  - Pass mock Page object")
    print("  - No actual browser launched")
    print("  - Fast test execution")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Browser Provider Pattern Examples")
    print("=" * 60 + "\n")
    
    example_1_default_provider()
    example_2_custom_browser_config()
    example_3_dependency_injection()
    example_4_remote_browser()
    example_5_persistent_context()
    example_6_shared_browser()
    example_7_factory_pattern()
    example_8_testing_setup()
    
    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)
