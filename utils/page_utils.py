"""
Page information and utility functions.
"""
from typing import TYPE_CHECKING
from playwright.sync_api import Page
from models import PageInfo

if TYPE_CHECKING:
    from action_executor import ScrollReason


class PageUtils:
    """Utilities for page information and basic operations"""
    
    def __init__(self, page: Page):
        self.page = page
        self.last_scroll_y = None
        self.last_scroll_x = None
    
    def get_page_info(self) -> PageInfo:
        """Get current page information"""
        try:
            viewport = self.page.viewport_size
            scroll_info = self.page.evaluate("""
                () => ({
                    scrollX: window.scrollX || 0,
                    scrollY: window.scrollY || 0
                })
            """)
            
            return PageInfo(
                width=viewport["width"],
                height=viewport["height"],
                scroll_x=int(scroll_info["scrollX"]),
                scroll_y=int(scroll_info["scrollY"]),
                url=self.page.url,
                title=self.page.title(),
                dpr=self.page.evaluate("window.devicePixelRatio"),
                ss_pixel_w=int(self.page.evaluate("window.innerWidth")),
                ss_pixel_h=int(self.page.evaluate("window.innerHeight")),
                css_scale=self.page.evaluate("window.devicePixelRatio"),
                doc_width=int(self.page.evaluate("document.body.scrollWidth")),
                doc_height=int(self.page.evaluate("document.body.scrollHeight"))
            )
        except Exception as e:
            print(f"⚠️ Error getting page info: {e}")
            # Return safe defaults
            return PageInfo(
                width=1280, height=800, scroll_x=0, scroll_y=0,
                url=self.page.url, title="",
                dpr=1.0,
                ss_pixel_w=1280,
                ss_pixel_h=800,
                css_scale=1.0,
                doc_width=1280,
                doc_height=800
            )
    
    def scroll_page(self, direction: str = "down", amount: int = 600, reason: "ScrollReason" = None, action_executor=None):
        """
        Scroll the page
        
        Args:
            direction: Direction to scroll ("down", "up")
            amount: Amount to scroll in pixels
            reason: Reason for scrolling (ScrollReason enum value)
            action_executor: Optional ActionExecutor to track scroll events
        """
        scroll_amount = amount if direction == "down" else -amount
        self.page.evaluate(f"window.scrollBy(0, {scroll_amount})")
        
        # Update tracked scroll position
        scroll_info = self.page.evaluate("""
            () => ({
                scrollX: window.scrollX || 0,
                scrollY: window.scrollY || 0
            })
        """)
        self.last_scroll_y = int(scroll_info["scrollY"])
        self.last_scroll_x = int(scroll_info["scrollX"])
        
        # Track the scroll event if action_executor is provided
        if action_executor and hasattr(action_executor, 'track_scroll_event') and reason is not None:
            action_executor.track_scroll_event(reason)
        
        import time
        time.sleep(0.5)
    
    def is_small_passive_scroll(self, current_scroll_y: int, current_scroll_x: int, threshold: int = 100) -> bool:
        """
        Detect if a small passive scroll occurred (browser-initiated, < threshold pixels).
        
        Args:
            current_scroll_y: Current Y scroll position
            current_scroll_x: Current X scroll position
            threshold: Pixel threshold for considering a scroll "small" (default: 100px)
            
        Returns:
            True if scroll changed by less than threshold pixels, False otherwise
        """
        # First check - no previous scroll to compare
        if self.last_scroll_y is None or self.last_scroll_x is None:
            self.last_scroll_y = current_scroll_y
            self.last_scroll_x = current_scroll_x
            return False
        
        y_delta = abs(current_scroll_y - self.last_scroll_y)
        x_delta = abs(current_scroll_x - self.last_scroll_x)
        
        # Update tracked position
        self.last_scroll_y = current_scroll_y
        self.last_scroll_x = current_scroll_x
        
        # Return True if the scroll is small (< threshold)
        return (y_delta > 0 or x_delta > 0) and y_delta < threshold and x_delta < threshold
