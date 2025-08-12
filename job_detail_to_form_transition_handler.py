#!/usr/bin/env python3
"""
Job Detail to Form Transition Handler

A handler that can take a job detail page and navigate autonomously to the form page.
This implements the algorithm from the flowchart to handle different page types and navigation flows.
"""

import base64
import time
import json
import traceback
from typing import Dict, Any, Optional, List, Tuple
from urllib.parse import urlparse
from playwright.sync_api import Page
from pydantic import BaseModel
from enum import Enum
import re
from google import genai
from page_detector import PageDetector, PageType
from typing import Optional, Dict, Any
import time
import json
from google.genai import types
import re

class PageState(Enum):
    """Different states/pages the handler can encounter"""
    LOGIN_WEBSITE = "login_website"
    VERIFICATION_WEBSITE = "verification_website"
    INPUT_NEEDED = "input_needed"
    JOB_APPLICATION_FORM = "job_application_form"
    APPLICATION_SUBMITTED = "application_submitted"
    ALREADY_APPLIED = "already_applied"
    UNKNOWN = "unknown"

class InteractionType(Enum):
    CLICK = "click"
    FILL_INPUT = "fill_input"
    TYPE = "type"
    
class SelectorValidation(BaseModel):
    valid: bool
    reason: str
    category: Optional[str] = None        # "forbidden" | "risky" | "ambiguous" | None
    suggestion: Optional[str] = None       # Optional improved selector

# --- New: static checks (deterministic) ---
_HARD_FORBIDDEN: List[Tuple[re.Pattern, str]] = [
    (re.compile(r'\biframe\b', re.I), "Cross-iframe selector detected. Build selectors as if already *inside* the iframe; do not include 'iframe' in the selector."),
    (re.compile(r'(^|[\s>])html\b', re.I), "Selector starts at <html>; overly generic and brittle."),
    (re.compile(r'(^|[\s>])body\b', re.I), "Selector starts at <body>; overly generic and brittle."),
    (re.compile(r'^\s*(div|span|label|form|section|article|li|ul|ol|table|tr|td|th|p|a)\s*$', re.I), "Selector is a bare tag with no discriminators; not robust."),
    (re.compile(r':nth-child\(\d+\)', re.I), "Positional selector ':nth-child()' is brittle."),
    (re.compile(r':nth-of-type\(\d+\)', re.I), "Positional selector ':nth-of-type()' is brittle."),
    (re.compile(r'\[hidden\]|\[aria-hidden\s*=\s*["\']?true["\']?\]', re.I), "Targets hidden UI; avoid hidden/non-interactive elements."),
    (re.compile(r'\.sr-only\b', re.I), "Targets screen-reader-only content; not interactable."),
    (re.compile(r'::-webkit-file-upload-button', re.I), "Pseudo-element is not a real node to click."),
    (re.compile(r'::shadow', re.I), "Shadow-piercing selectors are invalid/deprecated."),
    (re.compile(r':contains\(', re.I), "Non-standard ':contains()' pseudo-class."),
    (re.compile(r':visible\b', re.I), "Non-standard ':visible' pseudo-class."),
    (re.compile(r'\bcss\s*=', re.I), "Mixed engine syntax ('css=...') is brittle."),
    (re.compile(r'>>'), "Mixed engine selector combinator ('>>') is brittle."),
    (re.compile(r'^\s*//'), "XPath detected; use CSS or Playwright role/locator APIs instead."),
    (re.compile(r"a\s*\[\s*href\s*=\s*['\"]#['\"]\s*\]", re.I), "Void anchors with href='#' are not reliable controls."),
    (re.compile(r"a\s*\[\s*href\^=\s*['\"]javascript:", re.I), "javascript: links are non-standard; avoid."),
    (re.compile(r'\[class\^\s*=\s*["\']sc-'), "CSS-in-JS hashed classes change; unstable."),
    (re.compile(r'\[id\*\s*=\s*["\']__ember'), "Framework-generated IDs are volatile."),
    (re.compile(r'\[data-react(root|id)\]', re.I), "Internal React artifacts; unstable."),
    (re.compile(r'\[onclick\]'), "Behavior-based selection via inline handlers is unstable."),
    (re.compile(r'\.select__menu\b'), "Targets dropdown menu list, not the interactive control."),
    (re.compile(r'\.Toastify__toast\b'), "Ephemeral toast; not a control."),
    (re.compile(r'#root\s*>\s*:nth-child\(', re.I), "Deep positional chain; massively brittle."),
]

# "Risky" patterns: allowed only with strong scoping/uniqueness. We’ll ask Gemini to judge/fix.
_RISKY: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"^\s*text\s*=\s*['\"]", re.I), "Text selector alone is ambiguous; use role+name or scope to container."),
    (re.compile(r':nth-match\s*\(', re.I), "Engine-specific positional matching is brittle."),
    (re.compile(r'form\s+button\b', re.I), "Over-broad 'form button'; add scope (id/aria-label/legend)."),
    (re.compile(r'\[placeholder\s*=\s*["\']', re.I), "Placeholders localize; prefer label/role/name."),
    (re.compile(r'\.btn-primary\b', re.I), "Pure style class; unstable across themes."),
    (re.compile(r'\[aria-label\s*=\s*["\']Close["\']\s*\]$', re.I), "Multiple 'Close' buttons likely; scope to dialog."),
    (re.compile(r'img\s*\[\s*alt\s*=\s*["\']Submit["\']\s*\]\s*$', re.I), "Target the wrapping control, not the image."),
    (re.compile(r'svg\s*\[\s*aria-label\s*=\s*["\']Close["\']\s*\]\s*$', re.I), "Target the wrapping button, not SVG."),
    (re.compile(r'^button\s*\[\s*type\s*=\s*["\']submit["\']\s*\]\s*$', re.I), "Unscoped submit; ambiguous on multi-form pages."),
    (re.compile(r'^a\s*\[\s*role\s*=\s*["\']button["\']\s*\]\s*$', re.I), "Mixed semantics; ensure it’s the intended control and well-scoped."),
]

class InteractionInputType(Enum):
    TEXT = "text"
    TEXTAREA = "textarea"
    DATE = "date"
    FILE = "file"
    RADIO = "radio"
    CHECKBOX = "checkbox"
    DROPDOWN = "dropdown"
    DIV = "div"
    BUTTON = "button"
    LINK = "link"
    
class PageInteraction(BaseModel):
    """Base interaction"""
    interaction_type: InteractionType
    field_name: str
    element_selector: str
    input_type: InteractionInputType
    expected_value: str|None
    
class PageInteractionResponse(BaseModel):
    """Response to a page interaction"""
    interactions: List[PageInteraction]

class NavigationResult(BaseModel):
    """Result of navigation attempt"""
    success: bool
    current_state: PageState
    try_restart: bool = False
    next_url: Optional[str] = None
    error_message: Optional[str] = None
    requires_user_intervention: bool = False
    form_ready: bool = False
    already_applied: bool = False

class AdvanceButton(BaseModel):
    """Result of finding advance button"""
    button_text: Optional[str] = None
    button_selector: Optional[str] = None

class JobDetailToFormTransitionHandler:
    def __init__(self, page: Page, max_restarts: int = 3):
        self.page = page
        self.page_detector = PageDetector(page)
        self.max_restarts = max_restarts
        self.restart_count = 0
        self.current_state = PageState.UNKNOWN
        self.visited_urls = set()
        
    def click_accept_cookies_button(self) -> bool:
        """
        Click the accept cookies button
        
        Returns:
            bool: True if button was found and clicked successfully
        """
        try:
            print("🔍 Looking for accept cookies button...")
            
            # Common accept cookies button selectors
            accept_cookies_selectors = [
                'button:has-text("Accept all")',
                'button:has-text("Accept cookies")',
                'button:has-text("Accept")',
                'button:has-text("Allow all")',
                'button:has-text("Allow")',
                'button:has-text("Allow cookies")',
            ]
            
            accept_cookies_button = None
            for selector in accept_cookies_selectors:
                try:
                    accept_cookies_button = self.page.locator(selector)
                    if accept_cookies_button and accept_cookies_button.is_visible():
                        print(f"✅ Found accept cookies button with selector: {selector}")
                        break
                except:
                    continue
            
            if not accept_cookies_button:
                print("ℹ️ No accept cookies button found")
                return False
            
            # Click the accept cookies button
            print("🎯 Clicking accept cookies button...")
            accept_cookies_button.click()
            
            return True
        
        except Exception as e:
            print(f"❌ Error clicking accept cookies button: {e}")
            return False

    def navigate_to_form(self, preferences: Dict[str, Any], job_listing_url: str = None) -> NavigationResult:
        """
        Main method to navigate from job detail page to application form.
        
        Args:
            job_listing_url: Optional URL to navigate to if not already on job page
            
        Returns:
            NavigationResult: Information about the navigation attempt
        """
        try:
            print("🚀 Starting job detail to form navigation...")
            
            # Navigate to job listing if URL provided
            if job_listing_url:
                print(f"📍 Navigating to job listing: {job_listing_url}")
                self.page.goto(job_listing_url, wait_until='networkidle')
                time.sleep(2)
            
            # Main navigation loop
            while self.restart_count < self.max_restarts:
                print(f"\n🔄 Navigation iteration {self.restart_count + 1}/{self.max_restarts}")
                
                # Wait for 5 seconds
                time.sleep(5)
                
                # Detect current page state
                self.current_state = self._detect_page_state()
                print(f"📄 Current page state: {self.current_state.value}")
                
                # Handle the current state
                result = self._handle_current_state(preferences)
                
                if result.success:
                    if result.form_ready:
                        print("✅ Successfully navigated to application form!")
                        return result
                    elif result.requires_user_intervention:
                        print("⏸️ User intervention required, pausing...")
                        self.page.pause()
                        # After user intervention, continue
                        continue
                    elif result.already_applied:
                        print("✅ Already applied to this job")
                        return result
                    else:
                        # Continue to next state
                        continue
                else:
                    # Handle restart logic
                    if result.try_restart:
                        self.restart_count += 1
                        print(f"🔄 Restarting navigation (attempt {self.restart_count})")
                        continue
                    else:
                        print(f"❌ Navigation failed: {result.error_message}")
                        return result
            
            # Max restarts reached
            return NavigationResult(
                success=False,
                current_state=self.current_state,
                error_message=f"Max restarts ({self.max_restarts}) reached"
            )
            
        except Exception as e:
            print(f"❌ Error in navigation: {e}")
            return NavigationResult(
                success=False,
                current_state=self.current_state,
                error_message=str(e)
            )
    
    def _detect_page_state(self) -> PageState:
        """Detect the current page state using page detector and additional logic"""
        try:
            return self._detect_page_state_with_ai()
                
        except Exception as e:
            print(f"⚠️ Error detecting page state: {e}")
            return PageState.UNKNOWN
     
    def detect_and_handle_iframes(self) -> Dict[str, Any]:
        """
        Detect iframes on the page and determine if form fields are inside them
        
        Returns:
            Dict with iframe information and whether to use iframe context
        """
        try:
            print("🔍 Detecting iframes on the page...")
            
            # Find all iframes on the page
            iframes = self.page.query_selector_all('iframe')
            visible_iframes = [iframe for iframe in iframes if iframe.is_visible()]
            
            print(f"📋 Found {len(iframes)} total iframes, {len(visible_iframes)} visible")
            
            if not visible_iframes:
                print("ℹ️ No visible iframes found - using main page context")
                return {
                    'has_iframes': False,
                    'iframe_count': 0,
                    'use_iframe_context': False,
                    'iframe_context': None
                }
            
            # Check if any iframes contain form elements
            iframe_with_forms = None
            for i, iframe in enumerate(visible_iframes):
                try:
                    # Get iframe frame object
                    iframe_frame = iframe.content_frame()
                    if not iframe_frame:
                        continue
                    
                    # Check if iframe contains form elements
                    form_elements = iframe_frame.query_selector_all('input, select, textarea')
                    if form_elements:
                        print(f"✅ Found iframe {i+1} with {len(form_elements)} form elements")
                        iframe_with_forms = {
                            'index': i,
                            'iframe': iframe,
                            'frame': iframe_frame,
                            'form_count': len(form_elements)
                        }
                        break
                        
                except Exception as e:
                    print(f"⚠️ Error checking iframe {i+1}: {e}")
                    continue
            
            if iframe_with_forms:
                print(f"🎯 Using iframe context for form fields")
                return {
                    'has_iframes': True,
                    'iframe_count': len(visible_iframes),
                    'use_iframe_context': True,
                    'iframe_context': iframe_with_forms
                }
            else:
                print("ℹ️ No iframes with form elements found - using main page context")
                return {
                    'has_iframes': True,
                    'iframe_count': len(visible_iframes),
                    'use_iframe_context': False,
                    'iframe_context': None
                }
                
        except Exception as e:
            print(f"❌ Error detecting iframes: {e}")
            return {
                'has_iframes': False,
                'iframe_count': 0,
                'use_iframe_context': False,
                'iframe_context': None
            }
   
     
    def _take_smart_screenshot(self, full_page=True, frame=None, iframe_context=None):
        """
        Take a screenshot that includes iframe content if the form is in an iframe
        
        Args:
            frame: Optional iframe frame context. If None, uses main page.
            iframe_context: Optional iframe context dict with 'iframe' element info.
            
        Returns:
            Screenshot bytes and context info
        """
        try:
            # First priority: if we have iframe_context with iframe element, use it
            if iframe_context and isinstance(iframe_context, dict) and 'iframe' in iframe_context:
                print(f"📸 Opening iframe URL in new tab for screenshot")
                iframe_element = iframe_context['iframe']
                iframe_url = iframe_element.get_attribute('src')
                
                if not iframe_url:
                    print("⚠️ Iframe has no src URL, falling back to main page")
                    screenshot = self.page.screenshot(type="png", full_page=full_page)
                    context_str = "main page (no iframe src)"
                else:
                    # Make URL absolute if it's relative
                    if not iframe_url.startswith(('http://', 'https://')):
                        current_url = self.page.url
                        if iframe_url.startswith('/'):
                            # Absolute path from domain
                            parsed = urlparse(current_url)
                            iframe_url = f"{parsed.scheme}://{parsed.netloc}{iframe_url}"
                        else:
                            # Relative path
                            iframe_url = f"{current_url.rstrip('/')}/{iframe_url}"
                    
                    print(f"🔗 Opening iframe URL")
                    
                    # Store current page context
                    original_page = self.page
                    original_context = self.page.context
                    
                    try:
                        # Open iframe URL in new tab
                        new_page = original_context.new_page()
                        new_page.goto(iframe_url, wait_until='networkidle')
                        
                        # Take screenshot of the iframe content
                        screenshot = new_page.screenshot(type="png", full_page=full_page)
                        context_str = "iframe_content"
                        
                        print(f"✅ Successfully captured iframe content screenshot")
                        
                    finally:
                        # Always close the new tab and restore original page context
                        try:
                            new_page.close()
                        except Exception as close_error:
                            print(f"⚠️ Warning: Could not close iframe tab: {close_error}")
                        
                        # Restore original page context
                        self.page = original_page
                        
            # Second priority: if we have a frame with screenshot method, try to get iframe URL
            elif frame:
                # print(f"📸 Attempting to open iframe URL in new tab for better screenshot")
                
                # Try to find the iframe element that contains this frame
                iframe_element = None
                try:
                    # Get all iframes on the page
                    iframes = self.page.query_selector_all('iframe')
                    for iframe in iframes:
                        try:
                            if iframe.content_frame() == frame:
                                iframe_element = iframe
                                break
                        except:
                            continue
                except:
                    pass
                
                if iframe_element:
                    iframe_url = iframe_element.get_attribute('src')
                    if iframe_url:
                        # Make URL absolute if it's relative
                        if not iframe_url.startswith(('http://', 'https://')):
                            current_url = self.page.url
                            if iframe_url.startswith('/'):
                                # Absolute path from domain
                                parsed = urlparse(current_url)
                                iframe_url = f"{parsed.scheme}://{parsed.netloc}{iframe_url}"
                            else:
                                # Relative path
                                iframe_url = f"{current_url.rstrip('/')}/{iframe_url}"
                        
                        print(f"🔗 Opening iframe URL")
                        
                        # Store current page context
                        original_page = self.page
                        original_context = self.page.context
                        
                        try:
                            # Open iframe URL in new tab
                            new_page = original_context.new_page()
                            new_page.goto(iframe_url, wait_until='networkidle')
                            
                            # Take screenshot of the iframe content
                            screenshot = new_page.screenshot(type="png", full_page=True)
                            context_str = "iframe_content"
                            
                            print(f"✅ Successfully captured iframe content screenshot")
                            
                        finally:
                            # Always close the new tab and restore original page context
                            try:
                                new_page.close()
                            except Exception as close_error:
                                print(f"⚠️ Warning: Could not close iframe tab: {close_error}")
                            
                            # Restore original page context
                            self.page = original_page
                            print(f"🔄 Restored original page context")
                    else:
                        # Fallback to frame screenshot
                        print(f"⚠️ Iframe has no src URL, using frame screenshot")
                        screenshot = frame.screenshot(type="png", full_page=full_page)
                        context_str = "iframe_frame"
                else:
                    # Fallback to frame screenshot
                    print(f"⚠️ Could not find iframe element, using frame screenshot")
                    screenshot = frame.screenshot(type="png", full_page=full_page)
                    context_str = "iframe_frame"
                    
            # Third priority: if frame is a dict with 'frame' key
            elif frame and isinstance(frame, dict) and 'frame' in frame:
                print(f"📸 Taking screenshot of iframe content (extracted from dict)")
                actual_frame = frame['frame']
                if hasattr(actual_frame, 'screenshot'):
                    screenshot = actual_frame.screenshot(type="png", full_page=full_page)
                    context_str = "iframe"
                else:
                    raise Exception(f"Frame object does not have screenshot method: {type(actual_frame)}")
            else:
                # Take screenshot of main page
                print(f"📸 Taking screenshot of main page")
                screenshot = self.page.screenshot(type="png", full_page=full_page)
                context_str = "main page"
            
            # Save screenshot to file with context info
            filename = f"screenshot_{context_str.replace(' ', '_')}.png"
            with open(filename, "wb") as f:
                f.write(screenshot)
            
            return screenshot, context_str
            
        except Exception as e:
            print(f"❌ Error taking screenshot: {e}")
            # Fallback to main page screenshot
            try:
                screenshot = self.page.screenshot(type="png", full_page=full_page)
                with open("screenshot_fallback.png", "wb") as f:
                    f.write(screenshot)
                print(f"💾 Fallback screenshot saved: screenshot_fallback.png")
                return screenshot, "main page (fallback)"
            except Exception as fallback_error:
                print(f"❌ Critical error: Could not take any screenshot: {fallback_error}")
                return None, "error"

    def _take_screenshot_with_highlighted_elements(self, frame=None, iframe_context=None, highlight_selectors=None):
        """
        Take a screenshot with highlighted elements for debugging purposes
        
        Args:
            frame: Optional iframe frame context. If None, uses main page.
            iframe_context: Optional iframe context dict with 'iframe' element info.
            highlight_selectors: List of CSS selectors to highlight. If None, highlights ALL elements.
            
        Returns:
            Screenshot bytes and context info
        """
        try:
            # Determine which page context to use
            target_page = None
            target_frame = None
            
            if iframe_context and isinstance(iframe_context, dict) and 'iframe' in iframe_context:
                # Use iframe URL approach for better quality
                print(f"🎯 Opening iframe URL in new tab for highlighted screenshot")
                iframe_element = iframe_context['iframe']
                iframe_url = iframe_element.get_attribute('src')
                
                if iframe_url:
                    # Make URL absolute if it's relative
                    if not iframe_url.startswith(('http://', 'https://')):
                        current_url = self.page.url
                        if iframe_url.startswith('/'):
                            parsed = urlparse(current_url)
                            iframe_url = f"{parsed.scheme}://{parsed.netloc}{iframe_url}"
                        else:
                            iframe_url = f"{current_url.rstrip('/')}/{iframe_url}"
                    
                    # Store current page context
                    original_page = self.page
                    original_context = self.page.context
                    
                    try:
                        # Open iframe URL in new tab
                        new_page = original_context.new_page()
                        new_page.goto(iframe_url, wait_until='networkidle')
                        target_page = new_page
                        context_str = "iframe_content"
                    except Exception as e:
                        print(f"⚠️ Could not open iframe URL: {e}, falling back to frame")
                        target_frame = frame
                        context_str = "iframe_frame"
                else:
                    target_frame = frame
                    context_str = "iframe_frame"
                    
            elif frame:
                target_frame = frame
                context_str = "iframe_frame"
            else:
                target_page = self.page
                context_str = "main_page"
            
            # Default selectors to highlight if none provided
            if highlight_selectors is None:
                # Highlight ALL elements on the page with more specific selectors
                highlight_selectors = [
                    'input', 'select', 'textarea', 'button', 'a', 'div', 'span', 'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
                    'label', 'form', 'fieldset', 'legend', 'option', 'optgroup', 'ul', 'ol', 'li', 'table', 'tr', 'td', 'th'
                ]
            
            # Add highlighting styles to the page
            if target_page:
                print(f"🎯 Taking highlighted screenshot of page")
                # Add CSS for highlighting with stronger properties
                highlight_css = """
                <style id="element-highlighting">
                .highlighted-element {
                    outline: 4px solid #ff0000 !important;
                    outline-offset: 3px !important;
                    background-color: rgba(255, 0, 0, 0.2) !important;
                    position: relative !important;
                    z-index: 999999 !important;
                    box-shadow: 0 0 10px rgba(255, 0, 0, 0.8) !important;
                }
                .highlighted-element::after {
                    content: attr(data-selector);
                    position: absolute;
                    top: -30px;
                    left: 0;
                    background: #ff0000;
                    color: white;
                    padding: 4px 8px;
                    font-size: 12px;
                    font-family: monospace;
                    border-radius: 4px;
                    white-space: nowrap;
                    z-index: 1000000 !important;
                    font-weight: bold;
                    border: 2px solid white;
                }
                </style>
                """
                # Apply highlighting immediately using evaluate with arguments (avoid f-string brace issues)
                highlighted_count = target_page.evaluate(
                    """
                    (args) => {
                        const { highlightCss, selectors } = args;
                        // Add CSS
                        if (!document.getElementById('element-highlighting')) {
                            const style = document.createElement('style');
                            style.id = 'element-highlighting';
                            style.textContent = highlightCss;
                            document.head.appendChild(style);
                        }
                        // Add label CSS once
                        if (!document.getElementById('highlight-label-style')) {
                            const lblStyle = document.createElement('style');
                            lblStyle.id = 'highlight-label-style';
                            lblStyle.textContent = `
                                .highlight-label {
                                position: absolute;
                                background: #ff0000;
                                color: #ffffff;
                                padding: 2px 6px;
                                font-size: 11px;
                                font-family: monospace;
                                border-radius: 3px;
                                border: 2px solid #ffffff;
                                z-index: 1000001 !important;
                                pointer-events: none;
                                max-width: 420px;
                                overflow: hidden;
                                text-overflow: ellipsis;
                                white-space: nowrap;
                                }
                                `;
                            document.head.appendChild(lblStyle);
                        }
                        
                        // Highlight elements
                        let totalHighlighted = 0;
                        const cssEscape = (str) => (window.CSS && CSS.escape) ? CSS.escape(str) : String(str).replace(/([^a-zA-Z0-9_-])/g, '\\$1');
                        const getUniqueSelector = (element) => {
                            if (!element || element.nodeType !== 1) return '';
                            if (element.id) return '#' + cssEscape(element.id);
                            const parts = [];
                            let el = element;
                            while (el && el.nodeType === 1 && parts.length < 6) {
                                let selector = el.tagName.toLowerCase();
                                if (el.classList && el.classList.length > 0) {
                                    const className = Array.from(el.classList)[0];
                                    if (className) selector += '.' + cssEscape(className);
                                }
                                let siblingIndex = 1;
                                let prev = el.previousElementSibling;
                                while (prev) {
                                    if (prev.tagName === el.tagName) siblingIndex++;
                                    prev = prev.previousElementSibling;
                                }
                                selector += ':nth-of-type(' + siblingIndex + ')';
                                parts.unshift(selector);
                                if (el.id) { parts[0] = '#' + cssEscape(el.id); break; }
                                el = el.parentElement;
                                if (!el || el === document.body) break;
                            }
                            return parts.join(' > ');
                        };
                        selectors.forEach((selector) => {
                            try {
                                const elements = document.querySelectorAll(selector);
                                elements.forEach((el) => {
                                    el.classList.add('highlighted-element');
                                    const uniqueSelector = getUniqueSelector(el);
                                    el.setAttribute('data-selector', uniqueSelector || selector);
                                    totalHighlighted++;
                                });
                            } catch (e) {
                                console.log('Could not highlight selector:', selector, e);
                            }
                        });
                        // Remove any existing labels before placing new ones
                        document.querySelectorAll('.highlight-label').forEach(n => n.remove());
                        // Place non-overlapping labels
                        const placed = [];
                        const allHighlighted = Array.from(document.querySelectorAll('.highlighted-element'));
                        allHighlighted.forEach((el) => {
                            const uniqueSelector = el.getAttribute('data-selector') || '';
                            const rect = el.getBoundingClientRect();
                            const label = document.createElement('div');
                            label.className = 'highlight-label';
                            label.textContent = uniqueSelector;
                            // Initial position above element
                            let left = rect.left + window.scrollX;
                            let top = rect.top + window.scrollY - 22;
                            // Clamp within page width
                            const pageWidth = Math.max(document.documentElement.scrollWidth, document.body.scrollWidth);
                            label.style.left = left + 'px';
                            label.style.top = top + 'px';
                            document.body.appendChild(label);
                            let lr = label.getBoundingClientRect();
                            let moved = true;
                            let guard = 0;
                            while (moved && guard < 20) {
                                moved = false;
                                for (const r of placed) {
                                    const overlaps = !(lr.right < r.left || lr.left > r.right || lr.bottom < r.top || lr.top > r.bottom);
                                    if (overlaps) {
                                        top = r.bottom + 2;
                                        label.style.top = top + 'px';
                                        lr = label.getBoundingClientRect();
                                        moved = true;
                                    }
                                }
                                guard++;
                            }
                            // Clamp horizontally if needed (after layout)
                            if (lr.right > pageWidth) {
                                const newLeft = Math.max(0, pageWidth - lr.width - 4);
                                label.style.left = newLeft + 'px';
                                lr = label.getBoundingClientRect();
                            }
                            placed.push({ left: lr.left, top: lr.top, right: lr.right, bottom: lr.bottom });
                        });
                        
                        const highlightedElements = document.querySelectorAll('.highlighted-element');
                        return {
                            totalHighlighted: totalHighlighted,
                            highlightedClassCount: highlightedElements.length,
                            bodyChildren: document.body.children.length
                        };
                    }
                    """,
                    {"highlightCss": highlight_css, "selectors": highlight_selectors},
                )
                
                print(f"🎯 Highlighting applied: {highlighted_count}")
                
                # Wait a moment for highlighting to apply
                target_page.wait_for_timeout(1000)
                
                # Verify highlighting is still there before taking screenshot
                verification = target_page.evaluate("""
                    () => {
                        const highlighted = document.querySelectorAll('.highlighted-element');
                        const style = document.getElementById('element-highlighting');
                        return {
                            highlightedCount: highlighted.length,
                            styleExists: !!style,
                            bodyChildren: document.body.children.length
                        };
                    }
                """)
                print(f"🔍 Pre-screenshot verification: {verification}")
                
                # Take screenshot
                screenshot = target_page.screenshot(type="png", full_page=True)
                
                # Clean up highlighting
                target_page.evaluate("""
                    () => {
                        const highlighted = document.querySelectorAll('.highlighted-element');
                        highlighted.forEach((el) => {
                            el.classList.remove('highlighted-element');
                            el.removeAttribute('data-selector');
                        });
                        document.querySelectorAll('.highlight-label').forEach(n => n.remove());
                        const style = document.getElementById('element-highlighting');
                        if (style) style.remove();
                    }
                """)
                
            elif target_frame:
                print(f"🎯 Taking highlighted screenshot of frame")
                # For frames, we need to inject the highlighting into the frame context
                # but also ensure it's visible when taking a screenshot of the main page
                
                # First, get the iframe element's position and dimensions (Frame itself has no bounding_box)
                iframe_element_for_box = None
                try:
                    if iframe_context and isinstance(iframe_context, dict) and 'iframe' in iframe_context:
                        iframe_element_for_box = iframe_context.get('iframe')
                    if not iframe_element_for_box:
                        # Try to locate the iframe element by matching content_frame()
                        page_iframes = self.page.query_selector_all('iframe')
                        for _iframe_el in page_iframes:
                            try:
                                if _iframe_el.content_frame() == target_frame:
                                    iframe_element_for_box = _iframe_el
                                    break
                            except Exception:
                                continue
                except Exception:
                    iframe_element_for_box = None

                frame_box = None
                try:
                    if iframe_element_for_box:
                        frame_box = iframe_element_for_box.bounding_box()
                except Exception:
                    frame_box = None
                if not frame_box:
                    print("⚠️ Could not get frame bounding box, falling back to main page screenshot")
                    screenshot = self.page.screenshot(type="png", full_page=True)
                    context_str = "main_page_fallback"
                else:
                    print(f"📍 Frame position: x={frame_box['x']}, y={frame_box['y']}, width={frame_box['width']}, height={frame_box['height']}")
                    
                    # Prepare CSS strings to avoid quoting issues in JS
                    frame_highlight_css = (
                        ".highlighted-element {\n"
                        "  outline: 3px solid #ff0000 !important;\n"
                        "  outline-offset: 2px !important;\n"
                        "  background-color: rgba(255, 0, 0, 0.1) !important;\n"
                        "  position: relative !important;\n"
                        "  z-index: 10000 !important;\n"
                        "}\n"
                        ".highlighted-element::after { content: none !important; }\n"
                    )
                    label_css = (
                        ".highlight-label {\n"
                        "  position: absolute;\n"
                        "  background: #ff0000;\n"
                        "  color: #ffffff;\n"
                        "  padding: 2px 6px;\n"
                        "  font-size: 11px;\n"
                        "  font-family: monospace;\n"
                        "  border-radius: 3px;\n"
                        "  border: 2px solid #ffffff;\n"
                        "  z-index: 10001 !important;\n"
                        "  pointer-events: none;\n"
                        "  max-width: 420px;\n"
                        "  overflow: hidden;\n"
                        "  text-overflow: ellipsis;\n"
                        "  white-space: nowrap;\n"
                        "}\n"
                    )

                    # Try preferred method: open iframe src in a new tab and highlight there
                    try:
                        iframe_url = None
                        if iframe_element_for_box:
                            iframe_url = iframe_element_for_box.get_attribute('src')
                        if iframe_url:
                            if not iframe_url.startswith(('http://', 'https://')):
                                current_url = self.page.url
                                if iframe_url.startswith('/'):
                                    parsed = urlparse(current_url)
                                    iframe_url = f"{parsed.scheme}://{parsed.netloc}{iframe_url}"
                                else:
                                    iframe_url = f"{current_url.rstrip('/')}/{iframe_url}"

                            new_page = self.page.context.new_page()
                            new_page.goto(iframe_url, wait_until='networkidle')

                            # Inject highlighting and labels on the iframe page
                            new_page.evaluate(
                                """
                                (args) => {
                                    const { selectors, highlightCss, labelCss } = args;
                                    if (!document.getElementById('element-highlighting')) {
                                        const style = document.createElement('style');
                                        style.id = 'element-highlighting';
                                        style.textContent = highlightCss;
                                        document.head.appendChild(style);
                                    }
                                    if (!document.getElementById('highlight-label-style')) {
                                        const lblStyle = document.createElement('style');
                                        lblStyle.id = 'highlight-label-style';
                                        lblStyle.textContent = labelCss;
                                        document.head.appendChild(lblStyle);
                                    }
                                    let totalHighlighted = 0;
                                    const cssEscape = (str) => (window.CSS && CSS.escape) ? CSS.escape(str) : String(str).replace(/([^a-zA-Z0-9_-])/g, '\\$1');
                                    const getUniqueSelector = (element) => {
                                        if (!element || element.nodeType !== 1) return '';
                                        if (element.id) return '#' + cssEscape(element.id);
                                        const parts = [];
                                        let el = element;
                                        while (el && el.nodeType === 1 && parts.length < 6) {
                                            let selector = el.tagName.toLowerCase();
                                            if (el.classList && el.classList.length > 0) {
                                                const className = Array.from(el.classList)[0];
                                                if (className) selector += '.' + cssEscape(className);
                                            }
                                            let siblingIndex = 1;
                                            let prev = el.previousElementSibling;
                                            while (prev) {
                                                if (prev.tagName === el.tagName) siblingIndex++;
                                                prev = prev.previousElementSibling;
                                            }
                                            selector += ':nth-of-type(' + siblingIndex + ')';
                                            parts.unshift(selector);
                                            if (el.id) { parts[0] = '#' + cssEscape(el.id); return parts.join(' > '); }
                                            el = el.parentElement;
                                            if (!el || el === document.body) break;
                                        }
                                        return parts.join(' > ');
                                    };
                                    document.querySelectorAll('.highlight-label').forEach(n => n.remove());
                                    selectors.forEach((selector) => {
                                        try {
                                            const elements = document.querySelectorAll(selector);
                                            elements.forEach((el) => {
                                                el.classList.add('highlighted-element');
                                                const uniqueSelector = getUniqueSelector(el);
                                                el.setAttribute('data-selector', uniqueSelector || selector);
                                                totalHighlighted++;
                                            });
                                        } catch (e) {
                                            console.log('Could not highlight selector:', selector, e);
                                        }
                                    });
                                    const placed = [];
                                    const allHighlighted = Array.from(document.querySelectorAll('.highlighted-element'));
                                    allHighlighted.forEach((el) => {
                                        const uniqueSelector = el.getAttribute('data-selector') || '';
                                        const rect = el.getBoundingClientRect();
                                        const label = document.createElement('div');
                                        label.className = 'highlight-label';
                                        label.textContent = uniqueSelector;
                                        let left = rect.left + window.scrollX;
                                        let top = rect.top + window.scrollY - 20;
                                        const pageWidth = Math.max(document.documentElement.scrollWidth, document.body.scrollWidth);
                                        label.style.left = left + 'px';
                                        label.style.top = top + 'px';
                                        document.body.appendChild(label);
                                        let lr = label.getBoundingClientRect();
                                        let moved = true;
                                        let guard = 0;
                                        while (moved && guard < 20) {
                                            moved = false;
                                            for (const r of placed) {
                                                const overlaps = !(lr.right < r.left || lr.left > r.right || lr.bottom < r.top || lr.top > r.bottom);
                                                if (overlaps) {
                                                    top = r.bottom + 2;
                                                    label.style.top = top + 'px';
                                                    lr = label.getBoundingClientRect();
                                                    moved = true;
                                                }
                                            }
                                            guard++;
                                        }
                                        if (lr.right > pageWidth) {
                                            const newLeft = Math.max(0, pageWidth - lr.width - 4);
                                            label.style.left = newLeft + 'px';
                                            lr = label.getBoundingClientRect();
                                        }
                                        placed.push({ left: lr.left, top: lr.top, right: lr.right, bottom: lr.bottom });
                                    });
                                    return true;
                                }
                                """,
                                {"selectors": highlight_selectors, "highlightCss": frame_highlight_css, "labelCss": label_css},
                            )
                            new_page.wait_for_timeout(500)
                            screenshot = new_page.screenshot(type="png", full_page=True)
                            # Clean up on the new page
                            new_page.evaluate("() => { document.querySelectorAll('.highlight-label').forEach(n => n.remove()); const st = document.getElementById('element-highlighting'); if (st) st.remove(); }")
                            try:
                                new_page.close()
                            except Exception:
                                pass
                            context_str = "iframe_content_highlighted"
                        else:
                            raise Exception("No iframe src")
                    except Exception:
                        # Fallback: highlight inside frame and overlay frame bounds on main page
                        target_frame.evaluate(
                            """
                            (args) => {
                                const { selectors, highlightCss, labelCss } = args;
                                // Add CSS for highlighting
                                if (!document.getElementById('element-highlighting')) {
                                    const style = document.createElement('style');
                                    style.id = 'element-highlighting';
                                    style.textContent = highlightCss;
                                    document.head.appendChild(style);
                                }
                                // Add label CSS once
                                if (!document.getElementById('highlight-label-style')) {
                                    const lblStyle = document.createElement('style');
                                    lblStyle.id = 'highlight-label-style';
                                    lblStyle.textContent = labelCss;
                                    document.head.appendChild(lblStyle);
                                }
                                
                                // Highlight elements
                                let totalHighlighted = 0;
                                const cssEscape = (str) => (window.CSS && CSS.escape) ? CSS.escape(str) : String(str).replace(/([^a-zA-Z0-9_-])/g, '\\$1');
                                const getUniqueSelector = (element) => {
                                    if (!element || element.nodeType !== 1) return '';
                                    if (element.id) return '#' + cssEscape(element.id);
                                    const parts = [];
                                    let el = element;
                                    while (el && el.nodeType === 1 && parts.length < 6) {
                                        let selector = el.tagName.toLowerCase();
                                        if (el.classList && el.classList.length > 0) {
                                            const className = Array.from(el.classList)[0];
                                            if (className) selector += '.' + cssEscape(className);
                                        }
                                        let siblingIndex = 1;
                                        let prev = el.previousElementSibling;
                                        while (prev) {
                                            if (prev.tagName === el.tagName) siblingIndex++;
                                            prev = prev.previousElementSibling;
                                        }
                                        selector += ':nth-of-type(' + siblingIndex + ')';
                                        parts.unshift(selector);
                                        if (el.id) { parts[0] = '#' + cssEscape(el.id); break; }
                                        el = el.parentElement;
                                        if (!el || el === document.body) break;
                                    }
                                    return parts.join(' > ');
                                };
                                selectors.forEach((selector) => {
                                    try {
                                        const elements = document.querySelectorAll(selector);
                                        elements.forEach((el) => {
                                            el.classList.add('highlighted-element');
                                            const uniqueSelector = getUniqueSelector(el);
                                            el.setAttribute('data-selector', uniqueSelector || selector);
                                            totalHighlighted++;
                                        });
                                    } catch (e) {
                                        console.log('Could not highlight selector:', selector, e);
                                    }
                                });
                                // Remove any existing labels before placing new ones
                                document.querySelectorAll('.highlight-label').forEach(n => n.remove());
                                // Place non-overlapping labels in frame context
                                const placed = [];
                                const allHighlighted = Array.from(document.querySelectorAll('.highlighted-element'));
                                allHighlighted.forEach((el) => {
                                    const uniqueSelector = el.getAttribute('data-selector') || '';
                                    const rect = el.getBoundingClientRect();
                                    const label = document.createElement('div');
                                    label.className = 'highlight-label';
                                    label.textContent = uniqueSelector;
                                    let left = rect.left + window.scrollX;
                                    let top = rect.top + window.scrollY - 20;
                                    const pageWidth = Math.max(document.documentElement.scrollWidth, document.body.scrollWidth);
                                    label.style.left = left + 'px';
                                    label.style.top = top + 'px';
                                    document.body.appendChild(label);
                                    let lr = label.getBoundingClientRect();
                                    let moved = true;
                                    let guard = 0;
                                    while (moved && guard < 20) {
                                        moved = false;
                                        for (const r of placed) {
                                            const overlaps = !(lr.right < r.left || lr.left > r.right || lr.bottom < r.top || lr.top > r.bottom);
                                            if (overlaps) {
                                                top = r.bottom + 2;
                                                label.style.top = top + 'px';
                                                lr = label.getBoundingClientRect();
                                                moved = true;
                                            }
                                        }
                                        guard++;
                                    }
                                    if (lr.right > pageWidth) {
                                        const newLeft = Math.max(0, pageWidth - lr.width - 4);
                                        label.style.left = newLeft + 'px';
                                        lr = label.getBoundingClientRect();
                                    }
                                    placed.push({ left: lr.left, top: lr.top, right: lr.right, bottom: lr.bottom });
                                });
                                
                                console.log('Total elements highlighted in frame: ' + totalHighlighted);
                                return totalHighlighted;
                            }
                            """,
                            {"selectors": highlight_selectors, "highlightCss": frame_highlight_css, "labelCss": label_css},
                        )
                    
                    # Wait a moment for highlighting to apply
                    self.page.wait_for_timeout(500)
                    
                    # Also add a visible border around the frame itself on the main page
                    # so we can see where the frame is located
                    self.page.evaluate(f"""
                        () => {{
                            // Remove any existing frame highlight
                            const existing = document.getElementById('frame-highlight');
                            if (existing) existing.remove();
                            
                            // Add frame highlight
                            const frameHighlight = document.createElement('div');
                            frameHighlight.id = 'frame-highlight';
                            frameHighlight.style.cssText = `
                                position: absolute;
                                left: {frame_box['x']}px;
                                top: {frame_box['y']}px;
                                width: {frame_box['width']}px;
                                height: {frame_box['height']}px;
                                border: 4px solid #00ff00 !important;
                                background-color: rgba(0, 255, 0, 0.1) !important;
                                pointer-events: none;
                                z-index: 9999;
                            `;
                            
                            // Add label
                            frameHighlight.innerHTML = '<div style="position: absolute; top: -30px; left: 0; background: #00ff00; color: black; padding: 4px 8px; font-weight: bold; border-radius: 4px;">IFRAME CONTENT</div>';
                            
                            document.body.appendChild(frameHighlight);
                        }}
                    """)
                    
                    # Wait for frame highlight to be added
                    self.page.wait_for_timeout(200)
                    
                    # Take screenshot of the main page (which now includes both the frame highlight and the highlighted elements inside the frame)
                    screenshot = self.page.screenshot(type="png", full_page=True)
                    
                    # Clean up the frame highlight on the main page
                    self.page.evaluate("""
                        () => {
                            const frameHighlight = document.getElementById('frame-highlight');
                            if (frameHighlight) frameHighlight.remove();
                        }
                    """)
                    
                    # Clean up highlighting in the frame
                    target_frame.evaluate("""
                        () => {
                            const highlighted = document.querySelectorAll('.highlighted-element');
                            highlighted.forEach((el) => {
                                el.classList.remove('highlighted-element');
                                el.removeAttribute('data-selector');
                            });
                            document.querySelectorAll('.highlight-label').forEach(n => n.remove());
                            
                            const style = document.getElementById('element-highlighting');
                            if (style) style.remove();
                        }
                    """)
                    
                    context_str = "iframe_frame_highlighted"
            
            # Save highlighted screenshot to file
            filename = f"highlighted_screenshot_{context_str.replace(' ', '_')}.png"
            with open(filename, "wb") as f:
                f.write(screenshot)
            
            print(f"✅ Highlighted screenshot saved: {filename}")
            
            # Clean up if we opened a new page
            if target_page and target_page != self.page:
                try:
                    target_page.close()
                except Exception as close_error:
                    print(f"⚠️ Warning: Could not close highlighted screenshot tab: {close_error}")
                # Restore original page context
                self.page = original_page
            
            return screenshot, context_str
            
        except Exception as e:
            print(f"❌ Error taking highlighted screenshot: {e}")
            traceback.print_exc()
            # Fallback to regular screenshot
            try:
                return self._take_smart_screenshot(frame, iframe_context)
            except Exception as fallback_error:
                print(f"❌ Critical error: Could not take any screenshot: {fallback_error}")
                return None, "error"
 
    def _detect_page_state_with_ai(self, frame=None, iframe_context=None) -> PageState:
        """Use AI to detect page state when page detector is uncertain"""
        try:
            client = genai.Client(api_key="AIzaSyAU6PHwVlJJV5kogd4Es9hNf2Xy74fAOiA")
            
            page_content = self.page.content()
            screenshot, context_str = self._take_smart_screenshot(full_page=False, frame=frame, iframe_context=iframe_context)
            screenshot_part = types.Part.from_bytes(
                data=screenshot,
                mime_type="image/png"
            )
            prompt = f"""
            Analyze the page in the screenshot and classify it as exactly ONE of the following page types.
            
            Each type is **mutually exclusive**. 
            Select the most specific, primary function the page serves, based ONLY on what the user is clearly meant to do NOW.

            Possible page types and definitions (choose ONLY one, read each CAREFULLY):
            "login_website"
            "verification_website"
            "input_needed"
            "application_submitted"
            "already_applied"

            1. login_website
            - The only main actionable area is a form to SIGN IN or authenticate.
            - Features: username/email and password fields, “Sign In”/“Log in” buttons, SSO or OAuth (Google, LinkedIn) options.
            - **User must provide credentials to proceed**.
            - DO NOT select if the user can proceed without logging in, or if the page’s main action is verification or input of non-credential information.

            2. verification_website
            - The sole main action is to complete a security or identity check, NOT login.
            - Features: CAPTCHA (“I am not a robot”), SMS/email 2FA code entry, authentication app prompt, email link confirmation.
            - User cannot proceed without passing the verification.
            - **DO NOT select if the main page function is logging in, browsing jobs, or filling any application details**.

            3. input_needed
            - The ONLY way to proceed is to provide a single, non-sensitive, non-login, non-application piece of info (NOT a full form or credential).
            - Examples: “What is your location?”, “Are you eligible to work in the UK?”, “Please enter your date of birth”, checkbox consent.
            - The mini form could be in a modal or popup and will contain only a few fields.
            - DO NOT select if the page contains or leads to an application form, login, or verification.

            4. application_submitted
            - The MAIN page content (not just a modal) confirms a job application was successfully submitted.
            - Features: prominent message such as “Thank you for your application”, “Application received”, “We’ll be in touch”, or clear submission confirmation.
            - **The user is NOT asked for any more information**—this is a final, stand-alone confirmation screen.
            - This could be a modal or popup with an "Applied", "Submitted", "Received", "Thank you" message.
            - DO NOT select if the confirmation is only in a modal overlay (use “application_modal_form”), or if the page indicates you have already applied (use “already_applied”).

            5. already_applied
            - The page confirms that the user has ALREADY applied for the job, and CANNOT reapply.
            - Features: explicit message like “You have already applied for this job”, “Already applied”, “Application previously submitted”, or similar.
            - The user is NOT asked for further action or information.
            - DO NOT select if this is a generic confirmation (use “application_submitted”).

            Instructions:
            - Analyze ALL visible main content, forms, and user actions—IGNORE nav bars, ads, or unrelated widgets.
            - If the page fits more than one category, **pick the most specific, immediate user task (not the broadest context)**.
            - Return ONLY one of these strings, and NOTHING else:

            Allowed page types:
            "login_website"
            "verification_website"
            "input_needed"
            "application_submitted"
            "already_applied"
            "application_submitted_modal"

            HTML Content:
            {page_content}
            """
            
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=[
                    screenshot_part,
                    prompt],
                config=genai.types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_budget=32768),
                    system_instruction="You are a page type classifier. Return only the page type as a string."
                )
            )
            
            result = response.text.strip().lower()
            print(f"🔍 AI detected page type: {result}")
            
            # Map AI result to PageState
            state_mapping = {
                'login_website': PageState.LOGIN_WEBSITE,
                'verification_website': PageState.VERIFICATION_WEBSITE,
                'input_needed': PageState.INPUT_NEEDED,
                'job_application_form': PageState.JOB_APPLICATION_FORM,
                'application_submitted': PageState.APPLICATION_SUBMITTED,
                'already_applied': PageState.ALREADY_APPLIED,
            }
            
            return state_mapping.get(result, PageState.UNKNOWN)
            
        except Exception as e:
            print(f"⚠️ Error in AI page state detection: {e}")
            return PageState.UNKNOWN
    
    def clear_modal_if_present(self, modal_title: str = None, max_retries: int = 3):
        """Clear any modal if present with retry logic for API errors"""
        import time
        
        print("🔍 Checking for modal")
        
        for attempt in range(max_retries):
            try:
                
                client = genai.Client(api_key="AIzaSyAU6PHwVlJJV5kogd4Es9hNf2Xy74fAOiA")
                prompt = f"""
                Analyze the provided HTML of the current webpage and identify if there is a modal dialog currently present and visible to the user.  
                A modal is a popup, dialog, overlay, or window that prevents interaction with the main page until it is dismissed (e.g., login dialogs, cookie consent popups, alert overlays).

                Your objectives:
                1. Determine if a modal dialog is currently visible.
                2. If so, identify the single button, link, or element that closes the modal when clicked (e.g., "Close", "X", "Dismiss", "Cancel", etc).
                3. Return ONLY the most specific, stable, and reliable CSS selector for the close button.

                Guidelines for selector:
                - The close button will be in a modal with title {modal_title}. Ensure that the selector is with the modal title.
                - Selector must be valid, stable, and as specific as possible (use data-* attributes, IDs, unique classes, nth-of-type, etc).
                - Selector must directly target the clickable close element (never a parent container, never a generic button).
                - Do NOT use extremely generic selectors (like "button" or ".btn").
                - Do NOT return XPath or non-CSS selectors.
                - Do NOT include anything except the selector string in the response.
                - If no modal is present, return an empty string ("").
                - The close button is often an X icon or a close button in the modal

                Good selector examples:
                "button[data-testid='modal-close']"
                "button.close-modal"
                "div[role='dialog'] button[aria-label='Close']"
                "button#closeDialogButton"
                "a.modal__close"
                "span.icon-close"
                "button[class*='close']"
                "button[aria-label='Dismiss']"
                "div.cookie-modal button:nth-of-type(2)"

                Bad selector examples (do NOT return):
                "button"
                ".btn"
                ".close"      (if not unique on the page)
                "div"         (never target a non-clickable container)
                ".modal"      (the modal, not the button)
                "input"       (unless it's actually the close control)
                Any XPath selector
                Anything that matches multiple, unrelated elements

                Your output should be:
                - If a modal is present: the CSS selector for the close button, nothing else.
                - If no modal is present: return "" (an empty string).

                HTML of the page:
                {self.page.content()}
                """

                response = client.models.generate_content(
                    model="gemini-2.5-pro",
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        system_instruction="You are an expert at analyzing web pages and identifying modal elements.",
                    )
                )
                
                if response.text:
                    print(f"🔍 Attempting to clear modal (attempt {attempt + 1}/{max_retries})...")
                    
                    close_button_selector = response.text.replace("```json", "").replace("```", "").replace('"', "").strip()
                    close_button = self.page.query_selector(close_button_selector)
                    if close_button:
                        print(f"✅ Found close button: {close_button_selector}")
                        close_button.click()
                        return True
                    else:
                        # Try and click outside the modal
                        print("❌ No close button found, clicking outside the modal")
                        self.page.click("body")
                        return True
                else:
                    print("ℹ️ No modal detected")
                    return True

            except Exception as e:
                error_message = str(e)
                print(f"❌ Error clearing modal (attempt {attempt + 1}/{max_retries}): {error_message}")
                
                if attempt < max_retries - 1:
                    print(f"Retrying modal clearing (attempt {attempt + 1}/{max_retries})...")
                    wait_time = (attempt + 1) * 2  # Exponential backoff: 2s, 4s, 6s
                    time.sleep(wait_time)
                    continue
                else:
                    print("❌ Max retries reached for internal server error")
                    return self._fallback_modal_clear()
    
    def _fallback_modal_clear(self) -> bool:
        """Fallback method to clear modals when API fails"""
        try:
            print("🔄 Using fallback modal clearing method...")
            
            # Common modal close selectors to try
            common_close_selectors = [
                'button[aria-label*="Close"]',
                'button[aria-label*="Dismiss"]',
                'button[title*="Close"]',
                'button[title*="Dismiss"]',
                '.modal-close',
                '.modal__close',
                '.close-modal',
                '.dialog-close',
                '.popup-close',
                'button.close',
                'a.close',
                'span.close',
                'button[class*="close"]',
                'button[class*="dismiss"]',
                'button[class*="cancel"]',
                '.btn-close',
                '.btn-dismiss',
                '.btn-cancel',
                '[data-testid*="close"]',
                '[data-testid*="dismiss"]',
                'button:has-text("Close")',
                'button:has-text("Dismiss")',
                'button:has-text("Cancel")',
                'button:has-text("×")',
                'button:has-text("X")',
                'span:has-text("×")',
                'span:has-text("X")',
                'a:has-text("Close")',
                'a:has-text("Dismiss")'
            ]
            
            # Try each selector
            for selector in common_close_selectors:
                try:
                    element = self.page.query_selector(selector)
                    if element and element.is_visible():
                        print(f"✅ Found fallback close button: {selector}")
                        element.click()
                        time.sleep(1)  # Wait for modal to close
                        return True
                except Exception as e:
                    continue
            
            # If no close button found, try clicking outside the modal
            print("🔄 No close button found, trying to click outside modal...")
            try:
                # Click in the top-left corner of the page (usually outside modals)
                self.page.click("body", position={"x": 10, "y": 10})
                time.sleep(1)
                
                # Also try pressing Escape key
                self.page.keyboard.press("Escape")
                time.sleep(1)
                
                print("✅ Fallback modal clearing completed")
                return True
                
            except Exception as e:
                print(f"❌ Fallback modal clearing failed: {e}")
                return False
                
        except Exception as e:
            print(f"❌ Error in fallback modal clearing: {e}")
            return False
    
    def _handle_current_state(self, preferences) -> NavigationResult:
        """Handle the current page state and determine next action"""
        iframe_context = self.detect_and_handle_iframes()
        if iframe_context['use_iframe_context']:
            frame = iframe_context['iframe_context']['frame']
        else:
            frame = None
        
        try:
            if self.current_state == PageState.LOGIN_WEBSITE:
                return self._handle_login_website()
            elif self.current_state == PageState.VERIFICATION_WEBSITE:
                return self._handle_verification_website()
            elif self.current_state == PageState.INPUT_NEEDED:
                return self._handle_input_needed(preferences, frame, iframe_context)
            elif self.current_state == PageState.JOB_APPLICATION_FORM:
                return self._handle_job_application_form()
            elif self.current_state == PageState.APPLICATION_SUBMITTED:
                return NavigationResult(
                    success=True,
                    try_restart=False,
                    form_ready=False,
                    already_applied=True,
                    current_state=self.current_state
                )
            elif self.current_state == PageState.ALREADY_APPLIED:
                return NavigationResult(
                    success=True,
                    try_restart=False,
                    form_ready=False,
                    already_applied=True,
                    current_state=self.current_state
                )
            elif self.current_state == PageState.APPLICATION_SUBMITTED:
                return NavigationResult(
                    success=True,
                    try_restart=False,
                    form_ready=False,
                    current_state=self.current_state
                )
            else:
                return self._handle_unknown_state()
                
        except Exception as e:
            print(f"❌ Error handling current state: {e}")
            return NavigationResult(
                success=False,
                current_state=self.current_state,
                try_restart=True,
                error_message=str(e)
            )
           
    def _handle_job_board_website(self) -> NavigationResult:
        """Handle job board website - look for apply button and click next"""
        try:
            print("🔍 Looking for apply button on job board website...")
            
            # Find advance button
            advance_button = self._find_advance_button()
            context = self.page.context
            original_pages = set(context.pages) 
            
            if advance_button:
                print(f"✅ Found advance button: {advance_button.button_text}")
                
                # Click the apply button
                try:
                    button_element = self.page.query_selector(advance_button.button_selector)
                    button_element.click()
                    
                    # Check if navigated to new tab
                    # Wait for potential new tab
                    time.sleep(1)  # Replace with a smarter wait if possible

                    current_pages = set(context.pages)
                    new_pages = current_pages - original_pages

                    if new_pages:
                        # A new tab was opened
                        new_page = new_pages.pop()
                        self.page = new_page  # Switch to the new tab
                        print("🆕 Switched to new tab.")

                    # Check if we navigated to a new URL
                    new_url = self.page.url
                    if new_url not in self.visited_urls:
                        self.visited_urls.add(new_url)
                        print(f"📍 Navigated to: {new_url}")
                        
                    return NavigationResult(
                        success=True,
                        current_state=self.current_state,
                        next_url=new_url
                    )
                        
                except Exception as e:
                    print(f"❌ Error clicking apply button: {e}")
                    return NavigationResult(
                        success=False,
                        current_state=self.current_state,
                        try_restart=True,
                        error_message=f"Failed to click apply button: {e}"
                    )
            else:
                print("❌ No apply button found")
                return NavigationResult(
                    success=False,
                    current_state=self.current_state,
                    try_restart=True,
                    error_message="No apply button found on job board website"
                )
                
        except Exception as e:
            print(f"❌ Error handling job board website: {e}")
            return NavigationResult(
                success=False,
                current_state=self.current_state,
                try_restart=True,
                error_message=str(e)
            )
    
    def _handle_intermediary_website(self) -> NavigationResult:
        """Handle intermediary website - click next to continue"""
        try:
            print("🔍 Handling intermediary website...")
            
            # Look for continue/next buttons on intermediary pages
            advance_button = self._find_advance_button()
            context = self.page.context
            original_pages = set(context.pages) 
            
            if advance_button:
                print(f"✅ Found advance button: {advance_button.button_text}")
                
                try:
                    button_element = self.page.query_selector(advance_button.button_selector)
                    button_element.click()
                    
                    # Check if navigated to new tab
                    # Wait for potential new tab
                    time.sleep(1)  # Replace with a smarter wait if possible

                    current_pages = set(context.pages)
                    new_pages = current_pages - original_pages

                    if new_pages:
                        # A new tab was opened
                        new_page = new_pages.pop()
                        self.page = new_page  # Switch to the new tab
                        print("🆕 Switched to new tab.")

                    new_url = self.page.url
                    if new_url not in self.visited_urls:
                        self.visited_urls.add(new_url)
                        print(f"📍 Navigated to: {new_url}")
                    
                    return NavigationResult(
                        success=True,
                        current_state=self.current_state,
                        next_url=new_url
                    )
                        
                except Exception as e:
                    print(f"❌ Error clicking continue button: {e}")
                    return NavigationResult(
                        success=False,
                        current_state=self.current_state,
                        try_restart=True,
                        error_message=f"Failed to click continue button: {e}"
                    )
            else:
                # If no continue button, wait a bit and check if page auto-navigates
                print("⏳ No continue button found, waiting for auto-navigation...")
                time.sleep(5)
                
                new_url = self.page.url
                if new_url not in self.visited_urls:
                    self.visited_urls.add(new_url)
                    print(f"📍 Auto-navigated to: {new_url}")
                    
                    return NavigationResult(
                        success=True,
                        current_state=self.current_state,
                        try_restart=True,
                        next_url=new_url
                    )
                else:
                    return NavigationResult(
                        success=False,
                        current_state=self.current_state,
                        try_restart=True,
                        error_message="No continue button and no auto-navigation"
                    )
                    
        except Exception as e:
            print(f"❌ Error handling intermediary website: {e}")
            return NavigationResult(
                success=False,
                current_state=self.current_state,
                try_restart=True,
                error_message=str(e)
            )
    
    def _handle_login_website(self) -> NavigationResult:
        """Handle login website - defer to user"""
        
        print("🔒 Login website detected - deferring to user...")
        
        return NavigationResult(
            success=True,
            current_state=self.current_state,
            requires_user_intervention=True
        )
    
    def _handle_verification_website(self) -> NavigationResult:
        """Handle verification website (CAPTCHA, etc.) - defer to user"""
        
        print("🤖 Verification website detected - deferring to user...")
        
        return NavigationResult(
            success=True,
            current_state=self.current_state,
            requires_user_intervention=True
        )
    
    def _convert_screenshot_to_actions(self, frame=None, iframe_context=None) -> str:
        try:
            # Wait for 5 seconds
            time.sleep(5)
            
            client = genai.Client(api_key="AIzaSyAU6PHwVlJJV5kogd4Es9hNf2Xy74fAOiA")
            
            screenshot, context_info = self._take_smart_screenshot(full_page=True, frame=frame, iframe_context=iframe_context)
            screenshot_part = types.Part.from_bytes(
                data=screenshot,
                mime_type="image/png"
            )
            
            prompt = f"""
                You are given the screenshot of a webpage.
                Your task is to **analyze the page and describe in a single, clear paragraph the specific user actions required to progress the application.**
                Your instruction should be direct and not have alternatives. Just provide what
                You must not return a short answer, you must return a long comprehensive, detailed answer.

                Actions to describe (ONLY if they are clearly present and visible on the page):
                1. **Click** - Any visible, interactive element that must be clicked to proceed (buttons, links, icons, clickable divs, elements with role="button", etc.).
                2. **Fill input** - Any visible, enabled input field that must be filled, checked, or selected before proceeding. This includes:
                - Text fields: Describe what value should be typed, using the input’s label, placeholder, or aria-label (e.g., "type your email address in the 'Email' field").
                - Radio buttons and checkboxes: State which one to select or check, always using the visible label, group question, or their location for disambiguation.
                - Dropdowns (native or custom): Instruct the user to "select the best option from the dropdown," always referencing it by its visible label, placeholder, or clear location (e.g., "select the best option from the 'Country' dropdown at the top of the form").
                - File upload: If a visible upload button is present (not just an input), instruct the user to click the upload button (describe it by label, aria-label, or icon and location) and select their file.

                ---

                **Good Example 1:**  
                "To continue, type your email address in the 'Email' input field at the top of the form, select the best option from the 'Country' dropdown just below the email field, check the box labeled 'Subscribe to newsletter' at the bottom left, and then click the 'Next' button in the bottom right corner of the page."
                    - Identifies each input by label and location
                    - Dropdown is referenced with label and position, and the instruction is to select the best option

                **Good Example 2:**  
                "Begin by selecting the 'Yes' radio button beneath the eligibility question, then select the best option from the dropdown labeled 'Department' in the center of the page, and finally, click the circular icon with the arrow at the bottom right to proceed."
                    - Identifies each input by label and location
                    - Dropdown is referenced with label and position, and the instruction is to select the best option
                    - Radio button is referenced by label and location

                ---

                **Bad Example 1:**  
                "Choose an option from the dropdown and move on."
                    - Too vague; does not specify which dropdown or where

                **Bad Example 2:**  
                "Select the second option in the dropdown."
                    - Assumes which option to pick; never guess or invent the value

                **Bad Example 3:**  
                "Select from the dropdown with a blue border."
                    - References unreliable styling; not accessible or robust

                **Bad Example 4:**
                "click"
                    - Too vague; does not specify which element to click

                **Bad Example 5:**
                "click the button"
                    - Too vague; does not specify which button to click

                **Bad Example 6:**
                "SELECT_AND_CLICK"
                    - Too vague; does not specify which element to select and click

                **Bad Example 7:**
                "SELECT_AND_CLICK_AND_FILL"
                    - Too vague; does not specify which element to select and click and fill
                
                ---
                
                Your responses should be like the good examples, not the bad examples.
                """
            
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=[screenshot_part, prompt],
                config=genai.types.GenerateContentConfig(
                    system_instruction="You are an action determiner",
                    response_mime_type="text/plain"
                )
            )
            
            actions_paragraph = response.text.strip()
            
            return actions_paragraph
                    
        except Exception as e:
            print(f"❌ Error generating content: {e}")
            return ""
    
    def _normalize_selector(self, selector: Optional[str]) -> str:
        """Normalize selectors to Playwright-compatible syntax.
        - Convert :contains("text") to :has-text("text") anywhere in the selector
        """
        if not selector:
            return ""
        s = selector.strip()
        # Replace any :contains('...') or :contains("...") with :has-text("...")
        s = re.sub(r":contains\(\s*(['\"])\s*(.*?)\s*\1\s*\)", lambda m: f':has-text("{m.group(2)}")', s)
        return s

    def _convert_actions_to_interactions(self, text: str, preferences: Dict[str, Any], page_content: str, frame=None, iframe_context=None, selector_feedback: Optional[str] = None, attempt: int = 1, max_attempts: int = 2) -> PageInteractionResponse:
        """Convert a text description of actions to a list of interactions with one automatic retry on invalid selectors."""
        try:
            time.sleep(5)

            client = genai.Client(api_key="AIzaSyAU6PHwVlJJV5kogd4Es9hNf2Xy74fAOiA")
            screenshot, context_info = self._take_smart_screenshot(full_page=True, frame=frame, iframe_context=iframe_context)
            screenshot_part = genai.types.Part.from_bytes(data=screenshot, mime_type="image/png")

            base_prompt = f"""
                You are given a paragraph describing the actions that need to be taken to progress to the next page, a dictionary of user preferences, and the HTML content of the page.
                Use the screenshot to resolve ambiguity. If a modal is visible, focus ONLY on visible, enabled elements inside the modal.

                If provided, incorporate the FEEDBACK section strictly to correct prior mistakes.

                FEEDBACK (optional):
                {selector_feedback or "(none)"}

                [... the rest of your big prompt from before, unchanged ...]
                Paragraph:
                {text}

                Preferences:
                {json.dumps(preferences)}

                HTML content:
                {page_content}
                """

            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=[screenshot_part, base_prompt],
                config=genai.types.GenerateContentConfig(
                    system_instruction="You are an action converter. Return only the interactions as a list of JSON objects.",
                    response_mime_type="application/json",
                    response_schema=PageInteractionResponse
                )
            ).parsed

            # Normalize selectors to avoid invalid Playwright syntax (e.g., :contains)
            try:
                for inter in response.interactions:
                    inter.element_selector = self._normalize_selector(inter.element_selector)
            except Exception:
                pass

            # Validate selectors; if any invalid, retry once with feedback
            invalids: List[Tuple[int, str, str]] = []
            # for i, inter in enumerate(response.interactions):
            #     is_valid, reason = self._verify_interaction_is_valid(inter, page_content)
            #     if not is_valid:
            #         invalids.append((i, inter.element_selector, reason))

            # if invalids and attempt < max_attempts:
            #     feedback = self._format_selector_feedback(invalids)
            #     # Retry with feedback injected
            #     return self._convert_actions_to_interactions(
            #         text=text,
            #         preferences=preferences,
            #         page_content=page_content,
            #         frame=frame,
            #         iframe_context=iframe_context,
            #         selector_feedback=feedback,
            #         attempt=attempt + 1,
            #         max_attempts=max_attempts
            #     )

            return response

        except Exception as e:
            print(f"❌ Error converting actions: {e}")
            return PageInteractionResponse(interactions=[])
    
    def _static_selector_check(self, selector: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """Return (valid, reason, category). Category in {'forbidden','risky',None}."""
        s = (selector or "").strip()
        if not s:
            return False, "Empty selector.", "forbidden"

        for pat, msg in _HARD_FORBIDDEN:
            if pat.search(s):
                return False, msg, "forbidden"

        for pat, msg in _RISKY:
            if pat.search(s):
                return False, msg, "risky"

        return True, None, None
    
    def _llm_selector_validation(
        self,
        client: genai.Client,
        selector: str,
        page_content: str
    ) -> Tuple[bool, str, Optional[str], Optional[str]]:
        """
        Returns (valid, reason, category, suggestion). On any LLM error, degrades to (True, "LLM unavailable", None, None).
        """
        try:
            forbidden_summary = """
                Never use: iframe in selectors; html/body roots; bare tags without attributes; positional :nth-child/:nth-of-type; [hidden]/aria-hidden=true/.sr-only; ::-webkit-file-upload-button; ::shadow; :contains()/:visible; engine mixing (>>, css=); XPath;//; a[href="#"] or a[href^="javascript:"]; hashed framework classes/ids; [onclick]; .select__menu; .Toastify__toast; deep positional chains.
                Risky without scope: text='...'; :nth-match(...); 'form button'; [placeholder='...']; .btn-primary; [aria-label='Close'] alone; img[alt='Submit']; svg[aria-label='Close']; button[type='submit'] alone; a[role='button'] alone.
                """

            prompt = f"""
                Validate this selector against the rules below. If it violates, say why and propose a better selector if you can.

                Selector:
                {selector}

                Rules:
                {forbidden_summary}

                HTML (truncated as needed):
                {page_content[:120_000]}

                Return strictly JSON:
                {{
                "valid": true|false,
                "reason": "short, concrete reason naming the specific rule violated or why it's acceptable",
                "category": "forbidden" | "risky" | "ambiguous" | "ok",
                "suggestion": "improved selector if invalid or risky; otherwise empty"
                }}
                """

            out = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=[prompt],
                config=genai.types.GenerateContentConfig(
                    system_instruction="You are a strict selector validator. Output ONLY the JSON object per the schema.",
                    response_mime_type="application/json",
                    response_schema=SelectorValidation
                )
            ).parsed

            valid = bool(out.valid)
            reason = out.reason or ("OK" if valid else "Invalid selector.")
            category = out.category or ("ok" if valid else "ambiguous")
            suggestion = out.suggestion or None
            return valid, reason, category, suggestion

        except Exception:
            # Fail-open to avoid infinite loops; static checks already caught the worst issues.
            return True, "LLM unavailable; passed static checks.", None, None

    # --- Helper to format feedback for retry ---
    def _format_selector_feedback(self, invalids: List[Tuple[int, str, str]]) -> str:
        """
        invalids: list of (index, selector, reason)
        Returns a concise, line-separated feedback string for the prompt.
        """
        lines = [
            "Previous attempt included invalid or risky selectors. Fix them and do NOT repeat the same selectors.",
            "Problems:"
        ]
        for idx, sel, reason in invalids:
            lines.append(f"- Interaction {idx}: {reason}")
        return "\n".join(lines)

    def _verify_interaction_is_valid(self, interaction: "PageInteraction", page_content: str) -> Tuple[bool, str]:
        """
        Returns (is_valid, reason). 'reason' is a short, actionable message suitable for prompt feedback.
        """
        try:
            print(f"Verifying interaction: {interaction.element_selector}")
            selector = (interaction.element_selector or "").strip()
            ok, reason, category = self._static_selector_check(selector)
            if not ok:
                return False, f"{selector} — {reason}"

            # Static says it's fine or 'risky' — ask Gemini for a crisp decision and rationale.
            client = genai.Client(api_key="AIzaSyAU6PHwVlJJV5kogd4Es9hNf2Xy74fAOiA")  # consider using an env var
            valid, llm_reason, llm_category, suggestion = self._llm_selector_validation(client, selector, page_content)
            print(f"LLM validation: {valid}, {llm_reason}, {llm_category}, {suggestion}")
            
            if not valid:
                msg = f"{selector} — {llm_reason}"
                if suggestion:
                    msg += f" Suggested fix: {suggestion}"
                return False, msg

            return True, f"{selector} — OK"

        except Exception as e:
            # Fail-open with a log; don't block the pipeline.
            print(f"⚠️ Selector validation error: {e}")
            return True, "Validator error; allowing selector."

    
    def _is_element_in_form_context(self, element, frame=None) -> bool:
        """
        Check if an element is within the actual job application form context
        """
        try:
            if not element:
                return False
            
            context = frame if frame else self.page
            
            # Strategy 1: Check if element is within a form tag
            try:
                # Find the closest form ancestor
                form_ancestor = element.locator('xpath=ancestor::form').first
                if form_ancestor and form_ancestor.is_visible():
                    return True
            except:
                pass
            
            # Strategy 2: Check if element is within common job application form containers
            form_container_selectors = [
                'form',
                '[class*="application"]',
                '[class*="job-form"]',
                '[class*="apply"]',
                '[id*="application"]',
                '[id*="job-form"]',
                '[id*="apply"]',
                '.application-form',
                '.job-application',
                '.apply-form',
                '[role="form"]',
                '[data-testid*="application"]',
                '[data-testid*="form"]',
            ]
            
            for selector in form_container_selectors:
                try:
                    containers = context.query_selector_all(selector)
                    for container in containers:
                        if container.is_visible():
                            # Check if element is descendant of this container
                            try:
                                descendant = container.locator(f'#{element.get_attribute("id")}')
                                if descendant:
                                    return True
                            except:
                                pass
                            
                            # Check by comparing bounding boxes
                            try:
                                element_box = element.bounding_box()
                                container_box = container.bounding_box()
                                if element_box and container_box:
                                    # Check if element is within container bounds
                                    if (element_box['x'] >= container_box['x'] and
                                        element_box['y'] >= container_box['y'] and
                                        element_box['x'] + element_box['width'] <= container_box['x'] + container_box['width'] and
                                        element_box['y'] + element_box['height'] <= container_box['y'] + container_box['height']):
                                        return True
                            except:
                                pass
                except:
                    continue
            
            # Strategy 3: Check if element is NOT in common non-form areas
            non_form_selectors = [
                'nav',
                'header',
                'footer',
                '.navbar',
                '.navigation',
                '.menu',
                '.sidebar',
                '.filter',
                '.search-bar',
                '.breadcrumb',
                '[class*="nav"]',
                '[class*="header"]',
                '[class*="footer"]',
                '[class*="menu"]',
                '[class*="filter"]',
                '[class*="search"]',
                '[role="navigation"]',
                '[role="banner"]',
                '[role="contentinfo"]',
            ]
            
            for selector in non_form_selectors:
                try:
                    non_form_containers = context.query_selector_all(selector)
                    for container in non_form_containers:
                        try:
                            # Check if element is descendant of non-form container
                            descendant = container.locator(f'#{element.get_attribute("id")}')
                            if descendant:
                                return False
                        except:
                            pass
                except:
                    continue
            
            # Strategy 4: Enhanced iframe detection - check if iframe itself is form-related
            if frame:
                try:
                    # Check if iframe has form-related attributes
                    iframes = context.query_selector_all('iframe')
                    for iframe in iframes:
                        try:
                            iframe_src = iframe.get_attribute('src') or ''
                            iframe_id = iframe.get_attribute('id') or ''
                            iframe_class = iframe.get_attribute('class') or ''
                            
                            # Check for job application related keywords in iframe attributes
                            iframe_text = f"{iframe_src} {iframe_id} {iframe_class}".lower()
                            form_keywords = ['application', 'apply', 'job', 'career', 'form', 'greenhouse', 'workday', 'jobvite', 'lever']
                            
                            if any(keyword in iframe_text for keyword in form_keywords):
                                return True
                        except:
                            continue
                    
                    # If we can't determine iframe purpose, assume it's form-related
                    # (most job application sites use iframes for forms)
                    return True
                except:
                    return True
            
            # Strategy 5: Check for form-related keywords in surrounding context
            try:
                # Get parent elements and check for form-related text
                parent = element.locator('xpath=..').first
                if parent:
                    parent_text = parent.inner_text().lower()
                    form_keywords = ['application', 'apply', 'job', 'position', 'resume', 'cv', 'form', 'submit']
                    if any(keyword in parent_text for keyword in form_keywords):
                        return True
            except:
                pass
            
            # Default: If we can't determine, be conservative and allow it
            # (better to fill an extra field than miss a required one)
            return True
            
        except Exception as e:
            print(f"⚠️ Error checking form context: {e}")
            return True  # Default to allowing interaction
    
    def _is_system_or_preference_field(self, interaction: PageInteraction) -> bool:
        """
        Detect system or preference fields that should typically be skipped
        """
        field_name = interaction.field_name.lower() if interaction.field_name else ""
        field_selector = interaction.element_selector.lower() if interaction.element_selector else ""
        
        # Keywords that indicate system/preference fields
        system_keywords = [
            "theme", "color", "appearance", "display", "view", "sort", 
            "filter", "search", "preferences", "settings", "config",
            "notification", "alert", "email_pref", "communication",
            "privacy", "cookie", "tracking", "analytics", "marketing"
        ]
        
        # Check field content
        combined_text = f"{field_name} {field_selector}".lower()
        
        for keyword in system_keywords:
            if keyword in combined_text:
                return True
        
        # Check for pagination, sorting, or filtering controls
        if any(word in combined_text for word in ["per_page", "page_size", "sort_by", "order_by", "filter_by"]):
            return True
        
        # Check for fields that typically don't need user input in job applications
        non_essential_keywords = [
            "source", "referral", "utm", "campaign", "channel", "medium",
            "version", "build", "debug", "test", "demo"
        ]
        
        for keyword in non_essential_keywords:
            if keyword in combined_text:
                return True
        
        return False
    
    def _is_element_interactive(self, element, frame=None) -> bool:
        """
        Check if an element is interactive (visible, enabled, and in viewport)
        """
        try:
            if not element:
                return False
            
            # Check if element exists
            try:
                element.count()
            except:
                return False
            
            # Check if element is visible
            if not element.is_visible():
                return False
            
            # Check if element is enabled
            if not element.is_enabled():
                return False
            
            # Check if element is in viewport (not hidden by CSS)
            try:
                bounding_box = element.bounding_box()
                if not bounding_box:
                    return False
                
                # Check if element has zero dimensions (hidden)
                if bounding_box['width'] <= 0 or bounding_box['height'] <= 0:
                    return False
                
                # Check if element is positioned off-screen
                if (bounding_box['x'] < -1000 or bounding_box['y'] < -1000 or 
                    bounding_box['x'] > 10000 or bounding_box['y'] > 10000):
                    return False
                    
            except:
                # If we can't get bounding box, assume it's interactive
                pass
            
            # Check if element is not hidden by CSS
            try:
                display = element.evaluate("el => window.getComputedStyle(el).display")
                if display == 'none':
                    return False
                    
                visibility = element.evaluate("el => window.getComputedStyle(el).visibility")
                if visibility == 'hidden':
                    return False
                    
                opacity = element.evaluate("el => window.getComputedStyle(el).opacity")
                if opacity == '0':
                    return False
                    
            except:
                # If we can't check CSS properties, assume it's interactive
                pass
            
            return True
            
        except Exception as e:
            print(f"⚠️ Error checking element interactivity: {e}")
            return False
   
    def handle_click_interaction(self, interaction: PageInteraction, frame=None) -> bool:
        """Handle a click interaction"""
        try:
            print(f"🔄 Handling click interaction: {interaction.element_selector}")
            context = self.page.context
            original_pages = set(context.pages) 
            
            if interaction.element_selector.startswith('#'):
                if frame:
                    element = frame.locator(f'id={interaction.element_selector[1:]}').first
                else:
                    element = self.page.locator(f'id={interaction.element_selector[1:]}').first
            else:
                if frame:
                    element = frame.locator(interaction.element_selector)
                else:
                    element = self.page.locator(interaction.element_selector)
            
            if not element:
                print(f"⚠️ Element not found with selector: {interaction.element_selector}")
                return False
            
            # if not self._is_element_interactive(element, frame):
            #     context_str = "iframe" if frame else "page"
            #     print(f"⏭️ Skipping non-interactive element in {context_str}: {interaction.element_selector}")
            #     return True
            
            # if not self._is_element_in_form_context(element, frame):
            #     context_str = "iframe" if frame else "page"
            #     print(f"⏭️ Skipping element outside form context in {context_str}: {interaction.element_selector}")
            #     return True
            
            element.click()
            
            # Check if navigated to new tab
            # Wait for potential new tab
            time.sleep(1)  # Replace with a smarter wait if possible

            current_pages = set(context.pages)
            new_pages = current_pages - original_pages

            if new_pages:
                # A new tab was opened
                new_page = new_pages.pop()
                self.page = new_page  # Switch to the new tab
                print("🆕 Switched to new tab.")

            context_str = "iframe" if frame else "page"
            print(f"✅ Successfully clicked element in {context_str}: {interaction.element_selector}")
            
            # Check if in a new tab and switch to it
            
            return True
        except Exception as e:
            print(f"❌ Error handling click interaction: {e}")
            return False
    
    def handle_select_field(self, interaction: PageInteraction, frame=None) -> bool:
        """Handle a select field interaction"""
        try:
            print(f"🔄 Handling select field: {interaction.element_selector}")
            if interaction.element_selector.startswith('#'):
                if frame:
                    element = frame.locator(f'id={interaction.element_selector[1:]}').first
                else:
                    element = self.page.locator(f'id={interaction.element_selector[1:]}').first
            else:
                if frame:
                    element = frame.locator(interaction.element_selector)
                else:
                    element = self.page.locator(interaction.element_selector)
            
            # Check if element is interactive before attempting to interact
            if not self._is_element_interactive(element, frame):
                context_str = "iframe" if frame else "page"
                print(f"⏭️ Skipping non-interactive select field in {context_str}: {interaction.element_selector}")
                return True  # Return True to continue with other fields
            
            # Check if element is within form context before interacting
            if not self._is_element_in_form_context(element, frame):
                context_str = "iframe" if frame else "page"
                print(f"⏭️ Skipping select field outside form context in {context_str}: {interaction.element_selector}")
                return True
            
            # Check if this is a system or preference field that should be skipped
            if self._is_system_or_preference_field(interaction):
                context_str = "iframe" if frame else "page"
                print(f"⏭️ Skipping system/preference select field in {context_str}: {interaction.element_selector}")
                return True
            
            try:
                # Try standard select handling
                possible_values = element.locator("option").evaluate_all(
                    "options => options.map(option => option.value)"
                )
                
                alternative_value = self._find_alternative_select_value(possible_values, interaction, self.preferences)
                select_value = alternative_value
                
                if possible_values:
                    if alternative_value:
                        print(f"🔄 Using alternative value: {alternative_value}")
                        element.select_option(value=str(select_value))
                        context_str = "iframe" if frame else "page"
                        print(f"✅ Successfully selected option in {context_str}: {interaction.element_selector}")
                        return True
                    else:
                        print(f"❌ Could not find alternative value for {interaction.element_selector}")
                        return False
                else:
                    # Alternative: Click the select field and then click the option that is the closest match to the value
                    print("🔄 Clicking select field and then clicking the option that is the closest match to the value")
                    element.click()
                    time.sleep(0.5)
                    print(f"🔄 Clicked select field and waiting for options to appear")
                    
                    result = self._find_and_click_option_with_gpt(frame.content(), self.preferences, frame)
                    if result:
                        print(f"✅ Successfully selected option: {result}")
                        return True
                    else:
                        print(f"❌ Could not find option element with selector: {result}")
                        return False
            except Exception as e:
                print(f"❌ Error handling select field interaction: {e}")
                return False
        except Exception as e:
            print(f"❌ Error handling select field interaction: {e}")
            return False
        
    def handle_radio_interaction(self, interaction: PageInteraction, frame=None) -> bool:
        """Handle a radio interaction"""
        try:
            # Use id= selector engine for IDs with special characters like []
            if interaction.element_selector.startswith('#'):
                if frame:
                    element = frame.locator(f'id={interaction.element_selector[1:]}').first
                else:
                    element = self.page.locator(f'id={interaction.element_selector[1:]}').first
            else:
                if frame:
                    element = frame.locator(f"input[type='radio'][name='{interaction.field_name}'][value='{interaction.expected_value}']")
                else:
                    element = self.page.locator(f"input[type='radio'][name='{interaction.field_name}'][value='{interaction.expected_value}']")
            
            if not element:
                print(f"⚠️ Radio button not found with selector: {interaction.element_selector}")
                return False
            
            # if not self._is_element_interactive(element, frame):
            #     context_str = "iframe" if frame else "page"
            #     print(f"⏭️ Skipping non-interactive radio button in {context_str}: {interaction.element_selector}")
            #     return True
            
            radio_value: bool = interaction.expected_value
            if radio_value:
                element.check()
                context_str = "iframe" if frame else "page"
                print(f"✅ Clicked radio button in {context_str}")
            else:
                print(f"⏭️ Skipping radio button in {context_str}: {interaction.element_selector}")
            return True
        except Exception as e:
            print(f"❌ Error handling radio interaction: {e}")
            return False
        
    def handle_checkbox_interaction(self, interaction: PageInteraction, frame=None) -> bool:
        """Handle a checkbox interaction"""
        try:
            # Use id= selector engine for IDs with special characters like []
            if interaction.element_selector.startswith('#'):
                if frame:
                    element = frame.locator(f'id={interaction.element_selector[1:]}').first
                else:
                    element = self.page.locator(f'id={interaction.element_selector[1:]}').first
            else:
                if frame:
                    element = frame.locator(interaction.element_selector)
                else:
                    element = self.page.locator(interaction.element_selector)
            
            if not element:
                print(f"⚠️ Checkbox not found with selector: {interaction.element_selector}")
                return False
            
            if not self._is_element_interactive(element, frame):
                context_str = "iframe" if frame else "page"
                print(f"⏭️ Skipping non-interactive checkbox in {context_str}: {interaction.element_selector}")
                return True
            
            checkbox_value: bool = interaction.expected_value
            if checkbox_value:
                element.click()
                context_str = "iframe" if frame else "page"
                print(f"✅ Clicked checkbox in {context_str}")
                
            return True
        except Exception as e:
            print(f"❌ Error handling checkbox interaction: {e}")
            return False
    
    def handle_upload_interaction(self, interaction: PageInteraction, frame=None) -> bool:
        """Handle a upload interaction"""
        try:
            # Use id= selector engine for IDs with special characters like []
            if interaction.element_selector.startswith('#'):
                if frame:
                    element = frame.locator(f'id={interaction.element_selector[1:]}').first
                else:
                    element = self.page.locator(f'id={interaction.element_selector[1:]}').first
            else:
                if frame:
                    element = frame.locator(interaction.element_selector)
                else:
                    element = self.page.locator(interaction.element_selector)
            
            if not element:
                print(f"⚠️ Upload button not found with selector: {interaction.element_selector}")
                return False
                
            # Check if element is interactive before attempting to interact
            if not self._is_element_interactive(element, frame):
                context_str = "iframe" if frame else "page"
                print(f"⏭️ Skipping non-interactive upload button in {context_str}: {interaction.element_selector}")
                return True
                
            try:
                element.click()
                self.page.pause()
                
                context_str = "iframe" if frame else "page"
                print(f"✅ Successfully triggered file dialog in {context_str}")
                return True
            except Exception as e:
                print(f"⚠️ File dialog triggering failed: {e}")
                
                return False
        except Exception as e:
            print(f"❌ Error handling upload interaction: {e}")
            return False
        
    def handle_type_interaction(self, interaction: PageInteraction, frame=None) -> bool:
        """Handle a type interaction"""
        try:
            if interaction.element_selector.startswith('#'):
                if frame:
                    element = frame.locator(f'id={interaction.element_selector[1:]}').first
                else:
                    element = self.page.locator(f'id={interaction.element_selector[1:]}').first
            else:
                if frame:
                    element = frame.locator(interaction.element_selector)
                else:
                    element = self.page.locator(interaction.element_selector)
            
            if not element:
                print(f"⚠️ Element not found with selector: {interaction.element_selector}")
                return False
            
            # if not self._is_element_interactive(element, frame):
            #     context_str = "iframe" if frame else "page"
            #     print(f"⏭️ Skipping non-interactive element in {context_str}: {interaction.element_selector}")
            #     return True
            
            # if not self._is_element_in_form_context(element, frame):
            #     context_str = "iframe" if frame else "page"
            #     print(f"⏭️ Skipping element outside form context in {context_str}: {interaction.element_selector}")
            #     return True
            
            element.type(interaction.expected_value)
            return True
        except Exception as e:
            print(f"❌ Error handling type interaction: {e}")
            return False
    
    def handle_fill_input_interaction(self, interaction: PageInteraction, frame=None) -> bool:
        """Handle a fill input interaction"""
        try:
            if interaction.input_type == InteractionInputType.TEXT:
                self.handle_type_interaction(interaction, frame)
                return True
            elif interaction.input_type == InteractionInputType.DATE:
                # self.handle_type_interaction(interaction, frame)
                return True
            elif interaction.input_type == InteractionInputType.FILE:
                self.handle_upload_interaction(interaction, frame)
                return True
            elif interaction.input_type == InteractionInputType.RADIO:
                self.handle_radio_interaction(interaction, frame)
                return True
            elif interaction.input_type == InteractionInputType.CHECKBOX:
                self.handle_checkbox_interaction(interaction, frame)
                return True
            elif interaction.input_type == InteractionInputType.DROPDOWN:
                self.handle_select_field(interaction, frame)
                return True
            elif interaction.input_type == InteractionInputType.DIV:
                self.handle_click_interaction(interaction, frame)
                return True
            elif interaction.input_type == InteractionInputType.BUTTON:
                self.handle_click_interaction(interaction, frame)
                return True
            elif interaction.input_type == InteractionInputType.LINK:
                self.handle_click_interaction(interaction, frame)
                return True
        except Exception as e:
            print(f"❌ Error handling fill input interaction: {e}")
    
    def _handle_input_needed(self, preferences: Dict[str, Any], frame=None, iframe_context=None) -> NavigationResult:
        """Handle pages requiring specific input - get all inputs and fill them"""
        try:
            print("📝 Input needed page detected - getting all inputs...")

            # Figure out what needs to be done in the page
            print("Converting text to actions...")
            actions_paragraph = self._convert_screenshot_to_actions(frame=frame, iframe_context=iframe_context)
            print("Actions paragraph: ", actions_paragraph)
            
            print("Converting actions to interactions...")
            interactions = self._convert_actions_to_interactions(actions_paragraph, preferences, self.page.content(), frame=frame, iframe_context=iframe_context)
            print("Interactions found: ", len(interactions.interactions))
            print("Interactions: ", interactions.interactions)
            print("Filling inputs...")
            for interaction in interactions.interactions:
                if interaction.interaction_type == InteractionType.CLICK:
                    self.handle_click_interaction(interaction, frame)
                elif interaction.interaction_type == InteractionType.FILL_INPUT:
                    self.handle_fill_input_interaction(interaction, frame)
                elif interaction.interaction_type == InteractionType.TYPE:
                    self.handle_type_interaction(interaction, frame)
            
            return NavigationResult(
                success=True,
                current_state=self.current_state,
                form_ready=False
            )
        
        except Exception as e:
            print(f"❌ Error handling input needed: {e}")
            return NavigationResult(
                success=False,
                current_state=self.current_state,
                try_restart=True,
                error_message=str(e)
            )
    
    def _handle_job_application_form(self) -> NavigationResult:
        """Handle job application form - form is ready for filling"""
        
        print("📋 Job application form detected - ready for form filling!")
        
        return NavigationResult(
            success=True,
            current_state=self.current_state,
            form_ready=True
        )
    
    def _handle_unknown_state(self) -> NavigationResult:
        """Handle unknown page state - try to determine what to do"""
        print("❓ Unknown page state - attempting to determine action...")
        print("🔄 Retrying state detection")

        return NavigationResult(
            success=False,
            current_state=self.current_state,
            try_restart=True
        )
    
    def _find_advance_button(self) -> AdvanceButton:
        """Find advance button using AI"""
        try:
            client = genai.Client(api_key="AIzaSyAU6PHwVlJJV5kogd4Es9hNf2Xy74fAOiA")
            
            page_content = self.page.content()
            
            prompt = f"""
            You are given the HTML of a job application page.
            Your ONLY task is to identify the main button, within the currently active modal, dialog, or visible section, that will advance the job application process.
            This means the button that starts, continues, or submits the application (e.g., “Apply”, “Continue”, “Next”, “Submit Application”, etc).

            INSTRUCTIONS:
                •	Focus exclusively on the page section, modal, or dialog that is currently visible and active.
            • If multiple dialogs, modals, or overlays exist, pick the one with the highest z-index or that appears visually “on top.”
            • Ignore all elements outside the currently active context.
                •	Search for visible, clickable elements (typically <button>, <a>, or <input type="submit">) with text like:
                •	“Apply”, “Apply Now”, “Easy Apply”, “Quick Apply”
                •	“Submit”, “Submit Application”, “Start Application”
                •	“Continue”, “Next”, “Proceed”, “Go to next step”
                •	“Apply for this job”, “Apply to this position”
                •	Never return:
                •	Buttons like “Save”, “Cancel”, “Back”, “Share”, “Print”, “Download”, “Previous”, etc.
                •	Invisible or disabled buttons.
                •	If there are multiple candidates, choose the one whose text most clearly advances the application process, and which appears last in the sequence (e.g., “Continue” > “Next” > “Apply”).
                •	The CSS selector must:
                •	Point to the actual button element (not a container div)
                •	Be as specific and stable as possible (prefer IDs, data attributes, unique classes)
                •	NOT start with “iframe” unless the context is inside a relevant iframe
                •	NOT be empty, generic, or match multiple elements

            Return only a JSON object with these fields:
                •	text: The exact visible text on the button
                •	selector: The CSS selector for that button

            ⸻

            Good Example Responses:
            {{“text”: “Apply Now”, “selector”: “button[data-qa='apply-button']”}}
            {{“text”: “Submit Application”, “selector”: “button#submit-application”}}
            {{“text”: “Continue”, “selector”: “button.continue-btn”}}
            {{“text”: “Next”, “selector”: “button[aria-label='Next step']”}}

            Bad Example Responses:
            {{“text”: “Cancel”, “selector”: “button.cancel”}}      // 'Cancel' is not advancing
            {{“text”: “Apply”, “selector”: “div”}}                // Selector must target button, not container
            {{“text”: “Share”, “selector”: “.share-btn”}}         // 'Share' is unrelated
            {{“text”: “Apply Now”, “selector”: “iframe #apply”}}  // Do not use selectors starting with 'iframe' unless inside iframe context
            {{“text”: “Apply”, “selector”: “”}}                   // Selector is missing or blank
            {{“text”: “Next”, “selector”: “.btn”}}                // Selector is too generic and could match many buttons

            ⸻

            Return only the JSON object. Do not add explanations or any other text.

            HTML Content:
            {page_content}
            """
            
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction="You are an apply button finder. Return only valid JSON with text and selector fields.",
                    response_mime_type="application/json",
                    response_schema=AdvanceButton
                )
            )
            
            return response.parsed
            
        except Exception as e:
            print(f"⚠️ Error finding advance button: {e}")
            return None

        """Find submit button on input pages"""
        try:
            # Common submit button selectors
            submit_selectors = [
                'button:has-text("Submit")',
                'button:has-text("Continue")',
                'button:has-text("Next")',
                'input[type="submit"]',
                '[data-testid*="submit"]',
                '[data-testid*="continue"]'
            ]
            
            for selector in submit_selectors:
                try:
                    element = self.page.query_selector(selector)
                    if element and element.is_visible():
                        return {
                            'text': element.inner_text().strip() or element.get_attribute('value') or 'Submit',
                            'element': element,
                            'selector': selector
                        }
                except:
                    continue
            
            return None
            
        except Exception as e:
            print(f"⚠️ Error finding submit button: {e}")
            return None
   
        """Find any prominent button on the page"""
        try:
            # Look for prominent buttons
            button_selectors = [
                'button[class*="primary"]',
                'button[class*="main"]',
                'button[class*="submit"]',
                'button[class*="continue"]',
                'button[class*="next"]',
                'a[class*="primary"]',
                'a[class*="main"]',
                'a[class*="button"]'
            ]
            
            for selector in button_selectors:
                try:
                    element = self.page.query_selector(selector)
                    if element and element.is_visible():
                        return {
                            'text': element.inner_text().strip(),
                            'element': element,
                            'selector': selector
                        }
                except:
                    continue
            
            return None
            
        except Exception as e:
            print(f"⚠️ Error finding prominent button: {e}")
            return None