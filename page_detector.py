#!/usr/bin/env python3
"""
Page Detector Module

This module detects what type of page we're on and determines the best action to take.
"""

import base64
import traceback
from typing import Dict, Any
from enum import Enum
from urllib.parse import urlparse
from playwright.sync_api import Page
from google.genai import types

class PageType(Enum):
    """Enumeration of different page types"""
    SEARCH_PAGE = "search_page"           # Job search form page
    RESULTS_PAGE = "results_page"         # Job search results page
    JOB_DETAIL_PAGE = "job_detail_page"   # Individual job posting page
    APPLICATION_PAGE = "application_page" # Job application form page
    LOGIN_PAGE = "login_page"             # Login/authentication page
    CAPTCHA_PAGE = "captcha_page"         # CAPTCHA verification page
    ERROR_PAGE = "error_page"             # Error page (404, 500, etc.)
    UNKNOWN_PAGE = "unknown_page"         # Unknown page type

    @classmethod
    def get_page_type(cls, page_type: str) -> "PageType":
        if page_type == "search_page":
            return cls.SEARCH_PAGE
        elif page_type == "results_page":
            return cls.RESULTS_PAGE
        elif page_type == "job_detail_page":
            return cls.JOB_DETAIL_PAGE
        elif page_type == "application_page":
            return cls.APPLICATION_PAGE
        elif page_type == "login_page":
            return cls.LOGIN_PAGE
        elif page_type == "captcha_page":
            return cls.CAPTCHA_PAGE
        elif page_type == "error_page":
            return cls.ERROR_PAGE
        else:
            return cls.UNKNOWN_PAGE

    def __eq__(self, other):
        return self.value == other.value

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value
    
    def __hash__(self):
        return hash(self.value)
    
    
    
class PageDetector:
    def __init__(self, page: Page):
        self.page = page
        self._cache = {}  # Cache for detection results
        self._last_url = None
        self._last_detection = None
    
    def detect_page_type(self, frame=None, iframe_context=None) -> PageType:
        """Detect the type of the current page using multiple strategies"""
        try:
            current_url = self.page.url
            
            print("🔍 Detecting page type...")
            print(f"📍 URL: {current_url}")
            
            title = self.page.title()
            print(f"📄 Title: {title}")
            
            # Try detection methods in order of accuracy (most accurate first)
            detection_result = PageType.UNKNOWN_PAGE

            print("🔍 Detecting page type...")
            try:
                gemini_result = self._detect_with_gemini(current_url, title, frame, iframe_context)
                if gemini_result != PageType.UNKNOWN_PAGE:
                    detection_result = gemini_result    
                    print(f"  ✅ Gemini detection successful! {gemini_result.value}")
                else:
                    print("  ⚠️ Gemini detection and page type detection failed")
            except Exception as e:
                print(f"  ❌ Error in Gemini detection: {e}")
            
            # Cache the result
            self._last_url = current_url
            
            print(f"✅ Page type detected: {detection_result.value}")
            
            return detection_result
            
        except Exception as e:
            print(f"❌ Error detecting page type: {e}")
            return PageType.UNKNOWN_PAGE
    
    def _detect_with_gemini(self, page_url: str, title: str, frame=None, iframe_context=None) -> PageType:
        """Use Gemini 2.5 Pro to analyze page type based on URL and title"""
        try:
            from google import genai
            
            # Get some page content for context
            page_content = self.page.content()
            
            # Take a screenshot of the page
            screenshot, context_str = self._take_screenshot_with_highlighted_elements(frame, iframe_context)

            # screenshot_base64 = base64.b64encode(screenshot).decode('utf-8')
            screenshot_part = types.Part.from_bytes(
                data=screenshot,
                mime_type="image/png"
            )
            
            prompt = f"""
                You are an expert page-classifier. Read the supplied metadata, ignore all HTML markup, and return **one** label—exactly as written— matching the page.
                You are also given a screenshot of the page. Use the screenshot to help you determine the page type.
                
                ### Input
                • URL: {page_url}  
                • Title: {title}  
                • Content snippet (truncated): {page_content}

                ### Allowed output labels
                search_page            - Job-search form where users enter keywords, location, or filters  
                results_page           - List of multiple job postings (may span pages)  
                job_detail_page        - Single job posting with full description / “Apply” button  
                application_page       - Form that collects applicant data (name, CV upload, etc.). There must at least be a name input field to be considered an application page.  
                login_page             - Authentication page (login, register, forgot-password)  
                captcha_page           - CAPTCHA or bot-verification interstitial  
                error_page             - HTTP or application error (404, 403, 500, maintenance)  
                company_profile_page   - Employer overview (about, benefits, culture)  
                static_info_page       - Non-job static content (blog article, help/FAQ, contact, T&C)  
                unknown_page           - Doesn't fit any label or information is insufficient

                ### Output contract
                Return **only** the label on its own line with no quotes, no extra whitespace, and no additional text.

                Examples  
                search_page  
                results_page  
                job_detail_page  
                application_page  
                login_page  
                captcha_page  
                error_page  
                company_profile_page  
                static_info_page  
                unknown_page
                """
            
            client = genai.Client(api_key="AIzaSyAU6PHwVlJJV5kogd4Es9hNf2Xy74fAOiA")
            
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    screenshot_part,
                    prompt],
                config=genai.types.GenerateContentConfig(
                    system_instruction="You are an expert at analyzing webpages and determining page types for job search automation. Return only valid JSON.",
                    response_mime_type="application/json",
                    response_schema=PageType
                )
            )
            
            result = response.text.replace("```json", "").replace("```", "").replace('"', "").strip()
            
            return PageType.get_page_type(result)
            
        except Exception as e:
            print(f"❌ Error in Gemini detection: {e}")
            return PageType.UNKNOWN_PAGE
    
    def _get_recommended_action(self, page_type: PageType) -> str:
        """Determine the recommended action based on page type"""
        action_map = {
            PageType.SEARCH_PAGE: 'fill_search_form',
            PageType.RESULTS_PAGE: 'find_job_listings',
            PageType.JOB_DETAIL_PAGE: 'evaluate_and_apply',
            PageType.APPLICATION_PAGE: 'fill_application_form',
            PageType.LOGIN_PAGE: 'wait_for_user_login',
            PageType.CAPTCHA_PAGE: 'wait_for_user_captcha',
            PageType.ERROR_PAGE: 'handle_error',
            PageType.UNKNOWN_PAGE: 'wait_for_user'
        }
        
        return action_map.get(page_type, 'wait_for_user')
    
    def _take_smart_screenshot(self, frame=None, iframe_context=None):
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
                    screenshot = self.page.screenshot(type="png", full_page=True)
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
                        screenshot = frame.screenshot(type="png", full_page=True)
                        context_str = "iframe_frame"
                else:
                    # Fallback to frame screenshot
                    print(f"⚠️ Could not find iframe element, using frame screenshot")
                    screenshot = frame.screenshot(type="png", full_page=True)
                    context_str = "iframe_frame"
                    
            # Third priority: if frame is a dict with 'frame' key
            elif frame and isinstance(frame, dict) and 'frame' in frame:
                print(f"📸 Taking screenshot of iframe content (extracted from dict)")
                actual_frame = frame['frame']
                if hasattr(actual_frame, 'screenshot'):
                    screenshot = actual_frame.screenshot(type="png", full_page=True)
                    context_str = "iframe"
                else:
                    raise Exception(f"Frame object does not have screenshot method: {type(actual_frame)}")
            else:
                # Take screenshot of main page
                print(f"📸 Taking screenshot of main page")
                screenshot = self.page.screenshot(type="png", full_page=True)
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
                screenshot = self.page.screenshot(type="png", full_page=True)
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

 