"""
Deterministic page evaluation functions using JavaScript/DOM.
These replace the AI-based functions in condition_engine.py for the "page" route.

The "page" route should use deterministic JavaScript evaluation instead of AI/vision
to provide faster, more reliable, and cost-effective page state detection.
"""

from typing import Dict, Any, Optional, Union
from playwright.sync_api import Page
from .base import GoalContext


class PageFunctionRegistry:
    """Registry for deterministic page evaluation functions."""
    
    def __init__(self):
        self.functions: Dict[str, callable] = {}
        self._register_core_functions()
    
    def _register_core_functions(self):
        """Register the core deterministic page functions."""
        self.functions.update({
            # Page state functions
            "env.page.at_bottom": self._at_bottom,
            "env.page.at_top": self._at_top,
            "env.page.scroll_y": self._scroll_y,
            "env.page.scroll_x": self._scroll_x,
            "env.page.scroll_percentage": self._scroll_percentage,
            
            # Viewport functions
            "env.page.viewport_width": self._viewport_width,
            "env.page.viewport_height": self._viewport_height,
            "env.page.viewport_size": self._viewport_size,
            
            # Document functions
            "env.page.document_width": self._document_width,
            "env.page.document_height": self._document_height,
            "env.page.document_size": self._document_size,
            
            # Basic page info
            "env.page.url": self._url,
            "env.page.title": self._title,
            "env.page.ready_state": self._ready_state,
            
            # Scroll utilities
            "env.page.can_scroll_down": self._can_scroll_down,
            "env.page.can_scroll_up": self._can_scroll_up,
            "env.page.scroll_to_bottom": self._scroll_to_bottom,
            "env.page.scroll_to_top": self._scroll_to_top,
        })
    
    def register_function(self, name: str, func: callable):
        """Register a new page function dynamically."""
        self.functions[name] = func
    
    def get_function(self, name: str) -> Optional[callable]:
        """Get a registered function by name."""
        return self.functions.get(name)
    
    def list_functions(self) -> list:
        """List all registered function names."""
        return list(self.functions.keys())
    
    # Core page state functions
    def _at_bottom(self, page: Page, args: Dict[str, Any]) -> bool:
        """Check if page is scrolled to bottom using JavaScript."""
        threshold = int(args.get("threshold", 100))
        
        js_code = f"""
        (() => {{
            const scrollY = window.scrollY;
            const innerHeight = window.innerHeight;
            const scrollHeight = document.documentElement.scrollHeight;
            return (scrollY + innerHeight) >= (scrollHeight - {threshold});
        }})()
        """
        
        try:
            return bool(page.evaluate(js_code))
        except Exception:
            return False
    
    def _at_top(self, page: Page, args: Dict[str, Any]) -> bool:
        """Check if page is scrolled to top using JavaScript."""
        threshold = int(args.get("threshold", 20))
        
        js_code = f"""
        (() => {{
            return window.scrollY <= {threshold};
        }})()
        """
        
        try:
            return bool(page.evaluate(js_code))
        except Exception:
            return True
    
    def _scroll_y(self, page: Page, args: Dict[str, Any]) -> int:
        """Get current vertical scroll position."""
        js_code = "window.scrollY"
        try:
            return int(page.evaluate(js_code))
        except Exception:
            return 0
    
    def _scroll_x(self, page: Page, args: Dict[str, Any]) -> int:
        """Get current horizontal scroll position."""
        js_code = "window.scrollX"
        try:
            return int(page.evaluate(js_code))
        except Exception:
            return 0
    
    def _scroll_percentage(self, page: Page, args: Dict[str, Any]) -> float:
        """Get scroll percentage (0.0 to 1.0)."""
        js_code = """
        (() => {
            const scrollY = window.scrollY;
            const innerHeight = window.innerHeight;
            const scrollHeight = document.documentElement.scrollHeight;
            const maxScroll = scrollHeight - innerHeight;
            return maxScroll > 0 ? scrollY / maxScroll : 0;
        })()
        """
        try:
            return float(page.evaluate(js_code))
        except Exception:
            return 0.0
    
    # Viewport functions
    def _viewport_width(self, page: Page, args: Dict[str, Any]) -> int:
        """Get viewport width."""
        js_code = "window.innerWidth"
        try:
            return int(page.evaluate(js_code))
        except Exception:
            return 0
    
    def _viewport_height(self, page: Page, args: Dict[str, Any]) -> int:
        """Get viewport height."""
        js_code = "window.innerHeight"
        try:
            return int(page.evaluate(js_code))
        except Exception:
            return 0
    
    def _viewport_size(self, page: Page, args: Dict[str, Any]) -> Dict[str, int]:
        """Get viewport dimensions as dict."""
        js_code = """
        (() => {
            return {
                width: window.innerWidth,
                height: window.innerHeight
            };
        })()
        """
        try:
            result = page.evaluate(js_code)
            return {"width": int(result.get("width", 0)), "height": int(result.get("height", 0))}
        except Exception:
            return {"width": 0, "height": 0}
    
    # Document functions
    def _document_width(self, page: Page, args: Dict[str, Any]) -> int:
        """Get document width."""
        js_code = "document.documentElement.scrollWidth"
        try:
            return int(page.evaluate(js_code))
        except Exception:
            return 0
    
    def _document_height(self, page: Page, args: Dict[str, Any]) -> int:
        """Get document height."""
        js_code = "document.documentElement.scrollHeight"
        try:
            return int(page.evaluate(js_code))
        except Exception:
            return 0
    
    def _document_size(self, page: Page, args: Dict[str, Any]) -> Dict[str, int]:
        """Get document dimensions as dict."""
        js_code = """
        (() => {
            return {
                width: document.documentElement.scrollWidth,
                height: document.documentElement.scrollHeight
            };
        })()
        """
        try:
            result = page.evaluate(js_code)
            return {"width": int(result.get("width", 0)), "height": int(result.get("height", 0))}
        except Exception:
            return {"width": 0, "height": 0}
    
    # Basic page info
    def _url(self, page: Page, args: Dict[str, Any]) -> str:
        """Get current URL."""
        try:
            return page.url
        except Exception:
            return ""
    
    def _title(self, page: Page, args: Dict[str, Any]) -> str:
        """Get page title."""
        try:
            return page.title()
        except Exception:
            return ""
    
    def _ready_state(self, page: Page, args: Dict[str, Any]) -> str:
        """Get document ready state."""
        js_code = "document.readyState"
        try:
            return str(page.evaluate(js_code))
        except Exception:
            return "unknown"
    
    # Scroll utilities
    def _can_scroll_down(self, page: Page, args: Dict[str, Any]) -> bool:
        """Check if page can be scrolled down."""
        js_code = """
        (() => {
            const scrollY = window.scrollY;
            const innerHeight = window.innerHeight;
            const scrollHeight = document.documentElement.scrollHeight;
            return (scrollY + innerHeight) < scrollHeight;
        })()
        """
        try:
            return bool(page.evaluate(js_code))
        except Exception:
            return False
    
    def _can_scroll_up(self, page: Page, args: Dict[str, Any]) -> bool:
        """Check if page can be scrolled up."""
        js_code = "window.scrollY > 0"
        try:
            return bool(page.evaluate(js_code))
        except Exception:
            return False
    
    def _scroll_to_bottom(self, page: Page, args: Dict[str, Any]) -> bool:
        """Scroll to bottom of page."""
        try:
            page.evaluate("window.scrollTo(0, document.documentElement.scrollHeight)")
            return True
        except Exception:
            return False
    
    def _scroll_to_top(self, page: Page, args: Dict[str, Any]) -> bool:
        """Scroll to top of page."""
        try:
            page.evaluate("window.scrollTo(0, 0)")
            return True
        except Exception:
            return False


# Global registry instance
_page_function_registry = PageFunctionRegistry()


def get_page_function_registry() -> PageFunctionRegistry:
    """Get the global page function registry."""
    return _page_function_registry


def register_page_function(name: str, func: callable):
    """Register a new page function globally."""
    _page_function_registry.register_function(name, func)


def list_page_functions() -> list:
    """List all available page functions."""
    return _page_function_registry.list_functions()


# Example of how to register custom functions
def register_custom_functions():
    """Example of registering custom page functions."""
    
    def scroll_past_element(page: Page, args: Dict[str, Any]) -> bool:
        """Check if scrolled past a specific element."""
        element_selector = args.get("element", "")
        if not element_selector:
            return False
        
        js_code = f"""
        (() => {{
            const element = document.querySelector('{element_selector}');
            if (!element) return false;
            const rect = element.getBoundingClientRect();
            return rect.bottom <= window.innerHeight;
        }})()
        """
        
        try:
            return bool(page.evaluate(js_code))
        except Exception:
            return False
    
    def element_in_viewport(page: Page, args: Dict[str, Any]) -> bool:
        """Check if element is in viewport."""
        element_selector = args.get("element", "")
        if not element_selector:
            return False
        
        js_code = f"""
        (() => {{
            const element = document.querySelector('{element_selector}');
            if (!element) return false;
            const rect = element.getBoundingClientRect();
            return rect.top >= 0 && rect.left >= 0 && 
                   rect.bottom <= window.innerHeight && 
                   rect.right <= window.innerWidth;
        }})()
        """
        
        try:
            return bool(page.evaluate(js_code))
        except Exception:
            return False
    
    # Register custom functions
    register_page_function("env.page.scrolled_past_element", scroll_past_element)
    register_page_function("env.page.element_in_viewport", element_in_viewport)


# Auto-register custom functions
register_custom_functions()
