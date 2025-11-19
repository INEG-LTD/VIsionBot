"""Human-in-the-loop middleware for BrowserVisionBot."""

from middleware import Middleware, ActionContext
from typing import Any, Callable, Optional


class HumanInTheLoopMiddleware(Middleware):
    """
    Pause for human intervention on specific conditions.
    
    Example:
        >>> bot.use(HumanInTheLoopMiddleware(on_captcha=True))
        ⏸️  CAPTCHA detected. Solve it and press Enter...
    """
    
    def __init__(
        self,
        on_captcha: bool = True,
        on_error: bool = False,
        on_action: Optional[Callable[[ActionContext], bool]] = None
    ):
        """
        Initialize human-in-the-loop middleware.
        
        Args:
            on_captcha: Pause when CAPTCHA is detected
            on_error: Pause when errors occur
            on_action: Custom function to determine when to pause
        """
        self.on_captcha = on_captcha
        self.on_error_flag = on_error
        self.on_action = on_action
    
    def before_action(self, context: ActionContext) -> ActionContext:
        """Check if we should pause before action."""
        should_pause = False
        message = ""
        
        # Check for CAPTCHA
        if self.on_captcha:
            action_str = str(context.action_data).lower()
            if 'captcha' in action_str or 'recaptcha' in action_str:
                should_pause = True
                message = "CAPTCHA detected"
        
        # Check custom condition
        if self.on_action and self.on_action(context):
            should_pause = True
            message = "Custom condition triggered"
        
        # Pause if needed
        if should_pause:
            self._pause(message, context)
        
        return context
    
    def on_error(self, context: ActionContext, error: Exception) -> None:
        """Pause on error if configured."""
        if self.on_error_flag:
            self._pause(f"Error occurred: {error}", context)
    
    def _pause(self, message: str, context: ActionContext) -> None:
        """Pause and wait for user input."""
        print(f"\n⏸️  {message}")
        print(f"   Action: {context.action_type}")
        print(f"   Page: {context.bot.page.url if context.bot.page else 'N/A'}")
        
        try:
            input("   Press Enter to continue...")
        except (EOFError, KeyboardInterrupt):
            print("\n   Continuing...")
