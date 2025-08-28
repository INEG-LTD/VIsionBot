"""
Cookie Handler Module

This module provides comprehensive cookie banner detection and handling functionality
for web automation. It can detect various types of cookie consent banners and
automatically handle them by clicking appropriate accept/consent buttons.
"""

from typing import Dict
from playwright.sync_api import Page


class CookieHandler:
    """
    Handles cookie consent banners and GDPR notices automatically.
    
    This class provides methods to:
    - Detect cookie banners using DOM analysis
    - Find and click appropriate accept/consent buttons
    - Verify that cookie banners have been dismissed
    - Handle various types of cookie consent interfaces
    """
    
    def __init__(self, page: Page, max_attempts: int = 2):
        """
        Initialize the cookie handler.
        
        Args:
            page: Playwright Page object to interact with
            max_attempts: Maximum number of attempts to handle cookies
        """
        self.page = page
        self.max_attempts = max_attempts
    
    def handle_cookies_with_dom(self) -> bool:
        """
        Main method to handle cookies using DOM-based detection.
        
        Returns:
            True if cookies were successfully handled, False otherwise
        """
        for attempt in range(1, self.max_attempts + 1):
            self.page.wait_for_load_state("domcontentloaded")
            self.page.wait_for_timeout(500)
            
            cookie_info = self.page.evaluate("""
                () => {
                const cookieBanners = [];
                const acceptButtons = [];
                const vis = el => el && el.offsetWidth > 0 && el.offsetHeight > 0;
                const inVp = r => r.top < window.innerHeight && r.bottom > 0;
                const all = document.querySelectorAll('*');
                for (const el of all) {
                    if (!vis(el)) continue;
                    const r = el.getBoundingClientRect();
                    if (!inVp(r)) continue;
                    const txt = (el.textContent||'').toLowerCase();
                    const cls = (el.className||'').toLowerCase();
                    const id  = (el.id||'').toLowerCase();
                    if (txt.includes('cookie')||cls.includes('cookie')||id.includes('cookie')||
                        txt.includes('gdpr')  ||cls.includes('gdpr')  ||id.includes('gdpr')  ||
                        txt.includes('consent')||cls.includes('consent')||id.includes('consent')) {
                    cookieBanners.push({tag:el.tagName,id:el.id,className:el.className,text:el.textContent?.trim()||''});
                    }
                    if (['button','a','input'].includes(el.tagName.toLowerCase())) {
                    if (/(^|\\b)(accept|agree|allow|consent|got it|ok|continue|save preferences)(\\b|$)/i.test(txt)) {
                        acceptButtons.push({tag:el.tagName,id:el.id,className:el.className,text:el.textContent?.trim()||''});
                    }
                    }
                }
                return {cookieBanners, acceptButtons};
                }
            """)
            
            if cookie_info['cookieBanners'] or cookie_info['acceptButtons']:
                if self._handle_cookies_dom(cookie_info):
                    self.page.wait_for_timeout(300)
                    more = self.page.evaluate("""
                        () => {
                        const sel = ['[id*="cookie"]','[class*="cookie"]','[id*="gdpr"]','[class*="gdpr"]','[id*="consent"]','[class*="consent"]','[id*="banner"]','[class*="banner"]'];
                        for (const s of sel) {
                            const els = document.querySelectorAll(s);
                            for (const el of els) if (el.offsetWidth>0 && el.offsetHeight>0) return true;
                        }
                        return false;
                        }
                    """)
                    if not more:
                        return True
            else:
                return True
        return False
    
    def _handle_cookies_dom(self, cookie_info: Dict) -> bool:
        """
        Handles cookies using DOM-based approach. Returns True if cookies were handled.
        
        Args:
            cookie_info: Dictionary containing cookie banners and accept buttons info
            
        Returns:
            True if cookies were successfully handled, False otherwise
        """
        try:
            # First try to find and click accept buttons
            if cookie_info['acceptButtons']:
                print(f"Found {len(cookie_info['acceptButtons'])} accept buttons via DOM")
                
                # Sort buttons by preference (accept all > accept > agree > allow)
                sorted_buttons = sorted(cookie_info['acceptButtons'], 
                                      key=lambda x: self._get_button_priority(x['text']))
                
                for button in sorted_buttons:
                    try:
                        print(f"Attempting to click accept button: {button['text']}")
                        
                        # Try different selector strategies
                        selectors = []
                        if button['id']:
                            selectors.append(f"#{button['id']}")
                        if button['className']:
                            for cls in button['className'].split():
                                if cls.strip():
                                    selectors.append(f".{cls.strip()}")
                        
                        # Add text-based selector as fallback
                        text_content = button['text'].replace('"', '\\"').replace("'", "\\'")
                        selectors.append(f"button:has-text('{text_content}')")
                        selectors.append(f"a:has-text('{text_content}')")
                        
                        clicked = False
                        for selector in selectors:
                            try:
                                element = self.page.query_selector(selector)
                                if element and element.is_visible():
                                    element.click()
                                    print(f"Successfully clicked cookie accept button using selector: {selector}")
                                    clicked = True
                                    break
                            except Exception as e:
                                print(f"Failed to click with selector {selector}: {e}")
                                continue
                        
                        if clicked:
                            # Wait a bit and check if cookie banner disappeared
                            self.page.wait_for_timeout(2000)
                            if self._check_cookies_gone():
                                print("Cookie banner successfully dismissed")
                                return True
                            
                    except Exception as e:
                        print(f"Failed to click button {button['text']}: {e}")
                        continue
            
            # If no accept buttons or they didn't work, try to find and click cookie banners
            if cookie_info['cookieBanners']:
                print(f"Found {len(cookie_info['cookieBanners'])} cookie banners via DOM")
                
                for banner in cookie_info['cookieBanners']:
                    try:
                        # Try to find close/dismiss buttons within the banner
                        close_selectors = [
                            '[aria-label*="close"]', '[aria-label*="dismiss"]',
                            '[title*="close"]', '[title*="dismiss"]',
                            'button[class*="close"]', 'button[class*="dismiss"]',
                            'a[class*="close"]', 'a[class*="dismiss"]',
                            '.close', '.dismiss', '.x', '.cross'
                        ]
                        
                        for selector in close_selectors:
                            try:
                                element = self.page.query_selector(selector)
                                if element and element.is_visible():
                                    element.click()
                                    print(f"Successfully clicked close button using selector: {selector}")
                                    self.page.wait_for_timeout(2000)
                                    if self._check_cookies_gone():
                                        print("Cookie banner successfully dismissed")
                                        return True
                                    break
                            except Exception:
                                continue
                                
                    except Exception as e:
                        print(f"Failed to handle cookie banner: {e}")
                        continue
            
            return False
            
        except Exception as e:
            print(f"Error in DOM-based cookie handling: {e}")
            return False
    
    def _check_cookies_gone(self) -> bool:
        """
        Checks if cookie banners have been dismissed.
        
        Returns:
            True if no cookie banners are visible, False otherwise
        """
        try:
            # Quick DOM check
            cookie_check = self.page.evaluate("""
                () => {
                    const cookieSelectors = [
                        '[id*="cookie"]', '[class*="cookie"]', '[data-testid*="cookie"]',
                        '[id*="gdpr"]', '[class*="gdpr"]', '[data-testid*="gdpr"]',
                        '[id*="consent"]', '[class*="consent"]', '[data-testid*="consent"]'
                    ];
                    
                    for (const selector of cookieSelectors) {
                        try {
                            const elements = document.querySelectorAll(selector);
                            for (const el of elements) {
                                if (el.offsetWidth > 0 && el.offsetHeight > 0) {
                                    const text = el.textContent?.toLowerCase() || '';
                                    if (text.includes('cookie') || text.includes('gdpr') || text.includes('consent')) {
                                        return false; // Cookie banner still visible
                                    }
                                }
                            }
                        } catch (e) {
                            // Ignore invalid selectors
                        }
                    }
                    return true; // No cookie banners found
                }
            """)
            
            return cookie_check
            
        except Exception as e:
            print(f"Error checking if cookies are gone: {e}")
            return False
    
    def _get_button_priority(self, button_text: str) -> int:
        """
        Returns priority for button selection (lower number = higher priority).
        
        Args:
            button_text: Text content of the button
            
        Returns:
            Priority number (1 = highest priority, 10 = lowest priority)
        """
        text_lower = button_text.lower()
        
        if 'accept all' in text_lower:
            return 1
        elif 'accept' in text_lower:
            return 2
        elif 'agree' in text_lower:
            return 3
        elif 'allow' in text_lower:
            return 4
        elif 'consent' in text_lower:
            return 5
        elif 'got it' in text_lower or 'gotit' in text_lower:
            return 6
        elif 'ok' in text_lower:
            return 7
        elif 'continue' in text_lower:
            return 8
        elif 'i understand' in text_lower:
            return 9
        else:
            return 10
