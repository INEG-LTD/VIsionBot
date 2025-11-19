"""
Middleware system for BrowserVisionBot.

Provides a flexible middleware pattern for intercepting and modifying bot behavior.
Inspired by Express.js and FastAPI middleware patterns.

Example:
    >>> from middleware import LoggingMiddleware, CostTrackingMiddleware
    >>> bot = BrowserVisionBot(config=BotConfig())
    >>> bot.use(LoggingMiddleware())
    >>> bot.use(CostTrackingMiddleware(max_cost=1.00))
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from vision_bot import BrowserVisionBot


@dataclass
class ActionContext:
    """
    Context passed to middleware hooks.
    
    Contains information about the action being executed and allows
    middleware to modify behavior.
    """
    
    action_type: str
    """Type of action: 'click', 'type', 'llm_call', 'iteration', etc."""
    
    action_data: Dict[str, Any]
    """Data associated with the action"""
    
    bot: 'BrowserVisionBot'
    """Reference to the bot instance"""
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Metadata that middleware can use to pass data between hooks"""
    
    should_continue: bool = True
    """If False, action execution will be skipped"""
    
    modified_data: Optional[Dict[str, Any]] = None
    """If set, this data will be used instead of action_data"""
    
    cached_result: Optional[Any] = None
    """If set and should_continue=False, this will be returned as the result"""


class Middleware(ABC):
    """
    Base class for middleware.
    
    Middleware can intercept actions before and after execution,
    handle errors, and modify behavior.
    
    Example:
        >>> class MyMiddleware(Middleware):
        ...     def before_action(self, context):
        ...         print(f"Starting: {context.action_type}")
        ...         return context
        ...     
        ...     def after_action(self, context, result):
        ...         print(f"Completed: {context.action_type}")
        ...         return result
    """
    
    def before_action(self, context: ActionContext) -> ActionContext:
        """
        Called before action executes.
        
        Args:
            context: Action context
            
        Returns:
            Modified context (can modify should_continue, modified_data, etc.)
        """
        return context
    
    def after_action(self, context: ActionContext, result: Any) -> Any:
        """
        Called after action completes successfully.
        
        Args:
            context: Action context
            result: Result from the action
            
        Returns:
            Modified result (or original result)
        """
        return result
    
    def on_error(self, context: ActionContext, error: Exception) -> None:
        """
        Called when action fails.
        
        Args:
            context: Action context
            error: The exception that was raised
        """
        pass


class MiddlewareManager:
    """
    Manages the middleware chain.
    
    Executes middleware hooks in order (before) and reverse order (after).
    """
    
    def __init__(self):
        self.middlewares: List[Middleware] = []
    
    def use(self, middleware: Middleware) -> None:
        """
        Add middleware to the chain.
        
        Args:
            middleware: Middleware instance to add
        """
        self.middlewares.append(middleware)
    
    def execute_before(self, context: ActionContext) -> ActionContext:
        """
        Execute all before_action hooks in order.
        
        Args:
            context: Action context
            
        Returns:
            Modified context
        """
        for middleware in self.middlewares:
            context = middleware.before_action(context)
            if not context.should_continue:
                break
        return context
    
    def execute_after(self, context: ActionContext, result: Any) -> Any:
        """
        Execute all after_action hooks in reverse order.
        
        Args:
            context: Action context
            result: Result from action
            
        Returns:
            Modified result
        """
        for middleware in reversed(self.middlewares):
            result = middleware.after_action(context, result)
        return result
    
    def execute_on_error(self, context: ActionContext, error: Exception) -> None:
        """
        Execute all on_error hooks.
        
        Args:
            context: Action context
            error: The exception that was raised
        """
        for middleware in self.middlewares:
            try:
                middleware.on_error(context, error)
            except Exception:
                # Don't let error handlers crash
                pass


class CostLimitExceeded(Exception):
    """Raised when cost limit is exceeded."""
    pass


class OutsideWorkingHours(Exception):
    """Raised when action attempted outside working hours."""
    pass
