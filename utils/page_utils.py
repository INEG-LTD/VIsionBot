"""
Page information and utility functions.
"""
from playwright.sync_api import Page
from models import PageInfo


class PageUtils:
    """Utilities for page information and basic operations"""
    
    def __init__(self, page: Page):
        self.page = page
    
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
    
    def scroll_page(self, direction: str = "down", amount: int = 600):
        """Scroll the page"""
        scroll_amount = amount if direction == "down" else -amount
        self.page.evaluate(f"window.scrollBy(0, {scroll_amount})")
        import time
        time.sleep(0.5)
