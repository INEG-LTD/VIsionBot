"""
Page information and utility functions.
"""
from playwright.sync_api import Page
from models import PageInfo


class PageUtils:
    """Utilities for page information and basic operations"""
    
    def __init__(self, page: Page):
        self.page = page
        self.last_scroll_y = None
        self.last_scroll_x = None
    
    def set_page(self, page: Page) -> None:
        """Update internal page reference."""
        if not page or page is self.page:
            return
        self.page = page
        self.last_scroll_y = None
        self.last_scroll_x = None
    
    def get_page_info(self) -> PageInfo:
        """Get current page information"""
        try:
            viewport = self.page.viewport_size
            # Single evaluate round-trip instead of 7 separate CDP calls.
            info = self.page.evaluate("""
                () => ({
                    scrollX: window.scrollX || 0,
                    scrollY: window.scrollY || 0,
                    dpr: window.devicePixelRatio || 1,
                    innerW: window.innerWidth || 0,
                    innerH: window.innerHeight || 0,
                    docW: document.body ? document.body.scrollWidth : 0,
                    docH: document.body ? document.body.scrollHeight : 0
                })
            """)

            return PageInfo(
                width=viewport["width"],
                height=viewport["height"],
                scroll_x=int(info["scrollX"]),
                scroll_y=int(info["scrollY"]),
                url=self.page.url,
                title=self.page.title(),
                dpr=info["dpr"],
                ss_pixel_w=int(info["innerW"]),
                ss_pixel_h=int(info["innerH"]),
                css_scale=info["dpr"],
                doc_width=int(info["docW"]),
                doc_height=int(info["docH"]),
            )
        except Exception as e:
            try:
                from utils.event_logger import get_event_logger
                get_event_logger().system_error(f"Error getting page info: {e}")
            except Exception:
                pass
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
