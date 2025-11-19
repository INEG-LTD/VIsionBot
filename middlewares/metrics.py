"""Metrics collection middleware for BrowserVisionBot."""

import time
from middleware import Middleware, ActionContext
from typing import Any, Dict


class MetricsMiddleware(Middleware):
    """
    Collect performance metrics.
    
    Tracks:
    - Number of actions
    - Number of LLM calls
    - Number of errors
    - Total execution time
    - Average action time
    
    Example:
        >>> metrics = MetricsMiddleware()
        >>> bot.use(metrics)
        >>> # ... run bot ...
        >>> print(metrics.get_metrics())
    """
    
    def __init__(self):
        """Initialize metrics middleware."""
        self.metrics = {
            'actions': 0,
            'llm_calls': 0,
            'errors': 0,
            'total_time': 0.0,
            'action_times': []
        }
    
    def before_action(self, context: ActionContext) -> ActionContext:
        """Start timing action."""
        context.metadata['start_time'] = time.time()
        return context
    
    def after_action(self, context: ActionContext, result: Any) -> Any:
        """Record metrics after action."""
        # Calculate elapsed time
        if 'start_time' in context.metadata:
            elapsed = time.time() - context.metadata['start_time']
            self.metrics['total_time'] += elapsed
            self.metrics['action_times'].append(elapsed)
        
        # Increment counters
        self.metrics['actions'] += 1
        
        if context.action_type == 'llm_call':
            self.metrics['llm_calls'] += 1
        
        return result
    
    def on_error(self, context: ActionContext, error: Exception) -> None:
        """Count errors."""
        self.metrics['errors'] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get collected metrics.
        
        Returns:
            Dictionary of metrics
        """
        avg_time = (
            self.metrics['total_time'] / self.metrics['actions']
            if self.metrics['actions'] > 0
            else 0
        )
        
        return {
            **self.metrics,
            'average_action_time': avg_time
        }
    
    def print_summary(self) -> None:
        """Print metrics summary."""
        metrics = self.get_metrics()
        print("\n" + "=" * 60)
        print("📊 METRICS SUMMARY")
        print("=" * 60)
        print(f"Actions: {metrics['actions']}")
        print(f"LLM Calls: {metrics['llm_calls']}")
        print(f"Errors: {metrics['errors']}")
        print(f"Total Time: {metrics['total_time']:.2f}s")
        print(f"Average Action Time: {metrics['average_action_time']:.2f}s")
        print("=" * 60 + "\n")
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics = {
            'actions': 0,
            'llm_calls': 0,
            'errors': 0,
            'total_time': 0.0,
            'action_times': []
        }
