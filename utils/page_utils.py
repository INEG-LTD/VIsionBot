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
    
    def scroll_page(self, direction: str = "down", amount: int = 600, reason: "ScrollReason" = None, action_executor=None):
        """
        Scroll the page or foreground modal

        Args:
            direction: Direction to scroll ("down", "up")
            amount: Amount to scroll in pixels
            reason: Reason for scrolling (ScrollReason enum value)
            action_executor: Optional ActionExecutor to track scroll events
        """
        scroll_amount = amount if direction == "down" else -amount

        # Debug logging
        print(f"🔍 [PageUtils] Attempting to scroll {direction} by {amount}px (scroll_amount={scroll_amount})")

        # Try to find and scroll the foreground element
        try:
            scrolled_modal = self.page.evaluate("""
            (scrollAmount) => {{
                // Try multiple scroll methods on an element
                function tryScrollElement(el, amount) {{
                    if (!el) return false;

                    const scrollBefore = el.scrollTop;

                    // Method 1: scrollBy
                    el.scrollBy(0, amount);
                    if (el.scrollTop !== scrollBefore) return true;

                    // Method 2: scrollTop direct assignment
                    el.scrollTop = scrollBefore + amount;
                    if (el.scrollTop !== scrollBefore) return true;

                    // Method 3: Dispatch wheel event (for custom scroll handlers)
                    try {{
                        const wheelEvent = new WheelEvent('wheel', {{
                            deltaY: amount,
                            deltaMode: 0,
                            bubbles: true,
                            cancelable: true
                        }});
                        el.dispatchEvent(wheelEvent);
                        // Give it a moment to process
                        setTimeout(() => {{}}, 10);
                        if (el.scrollTop !== scrollBefore) return true;
                    }} catch (e) {{}}

                    return false;
                }}

                // Check if element has scrollable content (relaxed check)
                function hasScrollableContent(el) {{
                    if (!el) return false;
                    return el.scrollHeight > el.clientHeight + 1;
                }}

                // Find scrollable element walking up the tree
                function findScrollableAncestor(el) {{
                    let current = el;
                    const candidates = [];

                    while (current && current !== document.body && current !== document.documentElement) {{
                        if (hasScrollableContent(current)) {{
                            candidates.push(current);
                        }}
                        current = current.parentElement;
                    }}

                    return candidates;
                }}

                // Strategy 1: Element at center of viewport
                const centerX = window.innerWidth / 2;
                const centerY = window.innerHeight / 2;
                const elementAtCenter = document.elementFromPoint(centerX, centerY);

                if (elementAtCenter) {{
                    // Try all scrollable ancestors
                    const scrollableAncestors = findScrollableAncestor(elementAtCenter);

                    for (const ancestor of scrollableAncestors) {{
                        if (tryScrollElement(ancestor, scrollAmount)) {{
                            console.log('[Scroll] Success - viewport center ancestor:', ancestor.tagName, ancestor.className);
                            return true;
                        }}
                    }}
                }}

                // Strategy 2: Any element with high z-index that has scrollable content
                const allElements = Array.from(document.querySelectorAll('*'));
                const candidates = [];

                for (const el of allElements) {{
                    const style = window.getComputedStyle(el);

                    // Skip hidden
                    if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') {{
                        continue;
                    }}

                    // Must have scrollable content
                    if (!hasScrollableContent(el)) {{
                        continue;
                    }}

                    const zIndex = parseInt(style.zIndex) || 0;
                    const position = style.position;

                    // Prioritize fixed/absolute with any z-index, or any element with z-index > 0
                    if ((position === 'fixed' || position === 'absolute') || zIndex > 0) {{
                        const priority = (position === 'fixed' || position === 'absolute' ? 10000 : 0) + zIndex;
                        candidates.push({{ el, priority }});
                    }}
                }}

                // Sort by priority
                candidates.sort((a, b) => b.priority - a.priority);

                // Try each candidate
                for (const candidate of candidates) {{
                    if (tryScrollElement(candidate.el, scrollAmount)) {{
                        console.log('[Scroll] Success - z-index element:', candidate.el.tagName, candidate.el.className, 'priority:', candidate.priority);
                        return true;
                    }}
                }}

                console.log('[Scroll] No foreground scrollable found, falling back to page scroll');
                return false;
            }}
        """, scroll_amount)
            print(f"🔍 [PageUtils] Modal scroll result: {scrolled_modal}")
        except Exception as e:
            print(f"⚠️ [PageUtils] Error during modal scroll detection: {e}")
            scrolled_modal = False

        # If no modal was scrolled, fall back to scrolling the main page
        if not scrolled_modal:
            print(f"🔍 [PageUtils] Falling back to main page scroll")
            self.page.evaluate(f"window.scrollBy(0, {scroll_amount})")
        else:
            print(f"✅ [PageUtils] Successfully scrolled modal/foreground element")

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
