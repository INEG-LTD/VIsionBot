"""
Examples demonstrating the Middleware system.

This file shows how to use middlewares to add cross-cutting concerns
like logging, cost tracking, metrics, and more.
"""

from bot_config import BotConfig, ModelConfig, ExecutionConfig
from vision_bot import BrowserVisionBot
from middleware import Middleware, ActionContext
from middlewares import (
    LoggingMiddleware,
    CostTrackingMiddleware,
    MetricsMiddleware,
    HumanInTheLoopMiddleware,
    RetryMiddleware,
    CachingMiddleware
)
from typing import Any


def example_1_basic_logging():
    """Example 1: Basic logging middleware"""
    print("=" * 60)
    print("Example 1: Basic Logging")
    print("=" * 60)
    
    bot = BrowserVisionBot(config=BotConfig())
    bot.use(LoggingMiddleware())
    
    print("✓ Bot created with logging middleware")
    print("  - All actions will be logged to console")
    print()


def example_2_cost_tracking():
    """Example 2: Cost tracking and limiting"""
    print("=" * 60)
    print("Example 2: Cost Tracking")
    print("=" * 60)
    
    bot = BrowserVisionBot(config=BotConfig())
    bot.use(CostTrackingMiddleware(max_cost=1.00))
    
    print("✓ Bot created with cost tracking")
    print("  - Max cost: $1.00")
    print("  - Will raise CostLimitExceeded if exceeded")
    print()


def example_3_chaining_middlewares():
    """Example 3: Chaining multiple middlewares"""
    print("=" * 60)
    print("Example 3: Chaining Middlewares")
    print("=" * 60)
    
    bot = BrowserVisionBot(config=BotConfig())
    bot.use(LoggingMiddleware(verbose=False)) \
       .use(MetricsMiddleware()) \
       .use(CostTrackingMiddleware(max_cost=5.00))
    
    print("✓ Bot created with 3 middlewares (chained)")
    print("  - Logging (non-verbose)")
    print("  - Metrics collection")
    print("  - Cost tracking ($5.00 limit)")
    print()


def example_4_human_in_loop():
    """Example 4: Human-in-the-loop for CAPTCHAs"""
    print("=" * 60)
    print("Example 4: Human-in-the-Loop")
    print("=" * 60)
    
    bot = BrowserVisionBot(config=BotConfig())
    bot.use(HumanInTheLoopMiddleware(on_captcha=True))
    
    print("✓ Bot created with human-in-the-loop")
    print("  - Will pause when CAPTCHA detected")
    print("  - User can solve and press Enter to continue")
    print()


def example_5_metrics_collection():
    """Example 5: Collect and display metrics"""
    print("=" * 60)
    print("Example 5: Metrics Collection")
    print("=" * 60)
    
    metrics = MetricsMiddleware()
    bot = BrowserVisionBot(config=BotConfig())
    bot.use(metrics)
    
    print("✓ Bot created with metrics middleware")
    print("  - Tracks actions, LLM calls, errors, timing")
    print("  - Access via: metrics.get_metrics()")
    print("  - Print summary: metrics.print_summary()")
    print()


def example_6_retry_middleware():
    """Example 6: Automatic retry on failure"""
    print("=" * 60)
    print("Example 6: Retry Middleware")
    print("=" * 60)
    
    bot = BrowserVisionBot(config=BotConfig())
    bot.use(RetryMiddleware(max_retries=3, backoff=2.0))
    
    print("✓ Bot created with retry middleware")
    print("  - Max retries: 3")
    print("  - Exponential backoff: 2.0x")
    print("  - Automatically retries failed actions")
    print()


def example_7_caching():
    """Example 7: Cache LLM responses"""
    print("=" * 60)
    print("Example 7: Caching Middleware")
    print("=" * 60)
    
    cache = CachingMiddleware(max_cache_size=1000)
    bot = BrowserVisionBot(config=BotConfig())
    bot.use(cache)
    
    print("✓ Bot created with caching middleware")
    print("  - Max cache size: 1000")
    print("  - Caches LLM responses")
    print("  - Check stats: cache.get_stats()")
    print()


def example_8_custom_middleware():
    """Example 8: Custom middleware"""
    print("=" * 60)
    print("Example 8: Custom Middleware")
    print("=" * 60)
    
    class NotifySlackMiddleware(Middleware):
        """Custom middleware to notify Slack"""
        
        def after_action(self, context: ActionContext, result: Any) -> Any:
            if context.action_type == 'task_complete':
                print(f"📢 [Slack] Task completed: {result}")
            return result
    
    bot = BrowserVisionBot(config=BotConfig())
    bot.use(NotifySlackMiddleware())
    
    print("✓ Bot created with custom middleware")
    print("  - Sends Slack notification on task completion")
    print()


def example_9_conditional_middleware():
    """Example 9: Conditional execution"""
    print("=" * 60)
    print("Example 9: Conditional Middleware")
    print("=" * 60)
    
    class WorkingHoursMiddleware(Middleware):
        """Only allow actions during working hours"""
        
        def before_action(self, context: ActionContext) -> ActionContext:
            from datetime import datetime
            hour = datetime.now().hour
            
            if hour < 9 or hour > 17:
                print(f"⏰ Outside working hours ({hour}:00)")
                context.should_continue = False
            
            return context
    
    bot = BrowserVisionBot(config=BotConfig())
    bot.use(WorkingHoursMiddleware())
    
    print("✓ Bot created with working hours middleware")
    print("  - Only runs 9 AM - 5 PM")
    print("  - Blocks actions outside working hours")
    print()


def example_10_full_stack():
    """Example 10: Full middleware stack"""
    print("=" * 60)
    print("Example 10: Full Middleware Stack")
    print("=" * 60)
    
    # Create all middlewares
    metrics = MetricsMiddleware()
    cache = CachingMiddleware()
    cost_tracker = CostTrackingMiddleware(max_cost=10.00)
    
    # Create bot with full stack
    bot = BrowserVisionBot(config=BotConfig())
    bot.use(LoggingMiddleware(verbose=False)) \
       .use(metrics) \
       .use(cache) \
       .use(cost_tracker) \
       .use(RetryMiddleware(max_retries=3)) \
       .use(HumanInTheLoopMiddleware(on_captcha=True))
    
    print("✓ Bot created with full middleware stack:")
    print("  1. Logging (non-verbose)")
    print("  2. Metrics collection")
    print("  3. Response caching")
    print("  4. Cost tracking ($10.00 limit)")
    print("  5. Automatic retry (3x)")
    print("  6. Human-in-the-loop (CAPTCHA)")
    print()
    print("After running:")
    print("  - metrics.print_summary()")
    print("  - cache.get_stats()")
    print("  - cost_tracker.get_total_cost()")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Middleware System Examples")
    print("=" * 60 + "\n")
    
    example_1_basic_logging()
    example_2_cost_tracking()
    example_3_chaining_middlewares()
    example_4_human_in_loop()
    example_5_metrics_collection()
    example_6_retry_middleware()
    example_7_caching()
    example_8_custom_middleware()
    example_9_conditional_middleware()
    example_10_full_stack()
    
    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)
