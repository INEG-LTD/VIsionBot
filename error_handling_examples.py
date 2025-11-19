"""
Examples demonstrating the Error Handling system.

Shows how to use structured errors, recovery strategies, and error handling middleware.
"""

from bot_config import BotConfig, ErrorHandlingConfig
from vision_bot import BrowserVisionBot
from middlewares import ErrorHandlingMiddleware, LoggingMiddleware
from error_handling import (
    BotError,
    ElementNotFoundError,
    NetworkError,
    CaptchaDetectedError,
    ErrorSeverity,
    RecoveryStrategy
)


def example_1_basic_error_handling():
    """Example 1: Basic error handling with middleware"""
    print("=" * 60)
    print("Example 1: Basic Error Handling")
    print("=" * 60)
    
    config = BotConfig(
        error_handling=ErrorHandlingConfig(
            screenshot_on_error=True,
            max_retries=3,
            retry_delay=1.0
        )
    )
    
    bot = BrowserVisionBot(config=config)
    error_middleware = ErrorHandlingMiddleware(config.error_handling)
    
    bot.use(error_middleware)
    
    print("✓ Bot created with error handling")
    print("  - Screenshots on error: True")
    print("  - Max retries: 3")
    print("  - Retry delay: 1.0s")
    print()


def example_2_custom_error_types():
    """Example 2: Using custom error types"""
    print("=" * 60)
    print("Example 2: Custom Error Types")
    print("=" * 60)
    
    # Different error types have different recovery strategies
    errors = [
        ElementNotFoundError("Button not found"),
        NetworkError("Connection timeout"),
        CaptchaDetectedError("CAPTCHA detected")
    ]
    
    for error in errors:
        print(f"Error: {error.__class__.__name__}")
        print(f"  - Severity: {error.severity.value}")
        print(f"  - Recovery: {error.recovery_strategy.value}")
        print()


def example_3_error_context():
    """Example 3: Error context preservation"""
    print("=" * 60)
    print("Example 3: Error Context")
    print("=" * 60)
    
    from error_handling import ErrorContext
    
    context = ErrorContext(
        error_type="ElementNotFoundError",
        message="Submit button not found",
        page_url="https://example.com",
        page_title="Example Page",
        action_type="click",
        action_data={"selector": "#submit"}
    )
    
    print("Error context captured:")
    print(f"  - Type: {context.error_type}")
    print(f"  - Message: {context.message}")
    print(f"  - Page: {context.page_url}")
    print(f"  - Action: {context.action_type}")
    print(f"  - Timestamp: {context.timestamp}")
    print()


def example_4_error_recovery():
    """Example 4: Automatic error recovery"""
    print("=" * 60)
    print("Example 4: Error Recovery")
    print("=" * 60)
    
    config = BotConfig(
        error_handling=ErrorHandlingConfig(
            max_retries=3,
            retry_delay=1.0,
            retry_backoff=2.0,  # Exponential backoff
            abort_on_critical=True
        )
    )
    
    print("✓ Error recovery configured:")
    print("  - Retry attempts: 3")
    print("  - Initial delay: 1.0s")
    print("  - Backoff: 2.0x (1s, 2s, 4s)")
    print("  - Abort on critical: True")
    print()


def example_5_error_summary():
    """Example 5: Error summary and reporting"""
    print("=" * 60)
    print("Example 5: Error Summary")
    print("=" * 60)
    
    error_middleware = ErrorHandlingMiddleware()
    
    # Simulate some errors
    from error_handling import ErrorContext
    error_middleware.error_handler.errors = [
        ErrorContext("NetworkError", "Timeout"),
        ErrorContext("ElementNotFoundError", "Button not found"),
        ErrorContext("NetworkError", "Connection refused"),
    ]
    
    summary = error_middleware.get_error_summary()
    
    print("Error Summary:")
    print(f"  - Total errors: {summary['total_errors']}")
    print(f"  - Error counts: {summary['error_counts']}")
    print()


def example_6_full_stack():
    """Example 6: Full error handling stack"""
    print("=" * 60)
    print("Example 6: Full Error Handling Stack")
    print("=" * 60)
    
    config = BotConfig(
        error_handling=ErrorHandlingConfig(
            screenshot_on_error=True,
            screenshot_dir="error_screenshots",
            max_retries=3,
            retry_delay=2.0,
            retry_backoff=2.0,
            abort_on_critical=True
        )
    )
    
    bot = BrowserVisionBot(config=config)
    error_middleware = ErrorHandlingMiddleware(config.error_handling)
    
    bot.use(LoggingMiddleware(verbose=False)) \
       .use(error_middleware)
    
    print("✓ Full error handling stack:")
    print("  1. Logging middleware")
    print("  2. Error handling middleware")
    print()
    print("Features:")
    print("  ✓ Automatic screenshots")
    print("  ✓ Exponential backoff retry")
    print("  ✓ Error context preservation")
    print("  ✓ Recovery strategies")
    print("  ✓ Error summary reporting")
    print()


def example_7_custom_recovery():
    """Example 7: Custom recovery strategies"""
    print("=" * 60)
    print("Example 7: Custom Recovery Strategies")
    print("=" * 60)
    
    print("Recovery strategies available:")
    print("  - RETRY: Retry the action automatically")
    print("  - SKIP: Skip the action and continue")
    print("  - ABORT: Stop automation immediately")
    print("  - ASK_USER: Prompt user for decision")
    print("  - FALLBACK: Use fallback action")
    print()
    
    print("Example: CAPTCHA detected")
    print("  → Strategy: ASK_USER")
    print("  → Prompts: 'How should we proceed? (retry/skip/abort)'")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Error Handling System Examples")
    print("=" * 60 + "\n")
    
    example_1_basic_error_handling()
    example_2_custom_error_types()
    example_3_error_context()
    example_4_error_recovery()
    example_5_error_summary()
    example_6_full_stack()
    example_7_custom_recovery()
    
    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)
