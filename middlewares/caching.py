"""Caching middleware for BrowserVisionBot."""

import hashlib
import json
from middleware import Middleware, ActionContext
from typing import Any, Dict


class CachingMiddleware(Middleware):
    """
    Cache LLM responses to avoid redundant API calls.
    
    Example:
        >>> bot.use(CachingMiddleware())
        # Identical prompts will return cached results
    """
    
    def __init__(self, max_cache_size: int = 1000):
        """
        Initialize caching middleware.
        
        Args:
            max_cache_size: Maximum number of cached responses
        """
        self.cache: Dict[str, Any] = {}
        self.max_cache_size = max_cache_size
        self.hits = 0
        self.misses = 0
    
    def before_action(self, context: ActionContext) -> ActionContext:
        """Check cache before action."""
        # Only cache LLM calls
        if context.action_type != 'llm_call':
            return context
        
        # Generate cache key
        cache_key = self._generate_key(context.action_data)
        
        # Check if in cache
        if cache_key in self.cache:
            self.hits += 1
            context.cached_result = self.cache[cache_key]
            context.should_continue = False  # Skip LLM call
            print(f"💾 Cache hit (hits: {self.hits}, misses: {self.misses})")
        else:
            self.misses += 1
            context.metadata['cache_key'] = cache_key
        
        return context
    
    def after_action(self, context: ActionContext, result: Any) -> Any:
        """Cache result after action."""
        # Only cache LLM calls
        if context.action_type != 'llm_call':
            return result
        
        # Store in cache
        if 'cache_key' in context.metadata:
            cache_key = context.metadata['cache_key']
            
            # Evict oldest if cache is full
            if len(self.cache) >= self.max_cache_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            
            self.cache[cache_key] = result
        
        return result
    
    def _generate_key(self, action_data: Dict[str, Any]) -> str:
        """Generate cache key from action data."""
        # Create deterministic string from action data
        data_str = json.dumps(action_data, sort_keys=True)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache)
        }
    
    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
