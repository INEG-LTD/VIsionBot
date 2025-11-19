"""Cost tracking middleware for BrowserVisionBot."""

from middleware import Middleware, ActionContext, CostLimitExceeded
from typing import Any


class CostTrackingMiddleware(Middleware):
    """
    Track and limit LLM API costs.
    
    Raises CostLimitExceeded when total cost exceeds max_cost.
    
    Example:
        >>> bot.use(CostTrackingMiddleware(max_cost=1.00))
        # Raises CostLimitExceeded if cost > $1.00
    """
    
    def __init__(self, max_cost: float = 1.00, warn_at: float = 0.80):
        """
        Initialize cost tracking middleware.
        
        Args:
            max_cost: Maximum allowed cost in USD
            warn_at: Warn when cost reaches this fraction of max_cost
        """
        self.max_cost = max_cost
        self.warn_at = warn_at
        self.total_cost = 0.0
        self.warned = False
    
    def after_action(self, context: ActionContext, result: Any) -> Any:
        """Track cost after action."""
        # Extract cost from result if available
        if isinstance(result, dict) and 'cost' in result:
            cost = result['cost']
            self.total_cost += cost
            
            # Check if we should warn
            if not self.warned and self.total_cost >= (self.max_cost * self.warn_at):
                print(f"⚠️  Cost warning: ${self.total_cost:.4f} / ${self.max_cost:.2f}")
                self.warned = True
            
            # Check if we exceeded limit
            if self.total_cost > self.max_cost:
                raise CostLimitExceeded(
                    f"Cost ${self.total_cost:.4f} exceeds limit of ${self.max_cost:.2f}"
                )
        
        return result
    
    def get_total_cost(self) -> float:
        """Get total cost so far."""
        return self.total_cost
    
    def reset(self) -> None:
        """Reset cost tracking."""
        self.total_cost = 0.0
        self.warned = False
