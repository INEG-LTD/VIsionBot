#!/usr/bin/env python3
"""
Job Detail to Form Transition Handler

A handler that can take a job detail page and navigate autonomously to the form page.
This implements the algorithm from the flowchart to handle different page types and navigation flows.
"""

import base64
import time
import json
from typing import Dict, Any, Optional, List
from playwright.sync_api import Page
from pydantic import BaseModel
from enum import Enum
from google import genai
from page_detector import PageDetector, PageType
from typing import Optional, Dict, Any
import time
import json
from google.genai import types

class PageState(Enum):
    """Different states/pages the handler can encounter"""
    JOB_BOARD_WEBSITE = "job_board_website"
    INTERMEDIARY_WEBSITE = "intermediary_website"
    LOGIN_WEBSITE = "login_website"
    VERIFICATION_WEBSITE = "verification_website"
    INPUT_NEEDED = "input_needed"
    JOB_APPLICATION_FORM = "job_application_form"
    APPLICATION_MODAL_FORM = "application_modal_form"
    APPLICATION_SUBMITTED = "application_submitted"
    APPLICATION_SUBMITTED_MODAL = "application_submitted_modal"
    ALREADY_APPLIED = "already_applied"
    UNKNOWN = "unknown"

class InteractionType(Enum):
    CLICK = "click"
    FILL_INPUT = "fill_input"
    TYPE = "type"

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

    def navigate_to_form(self, job_listing_url: str = None) -> NavigationResult:
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
                
                # Detect current page state
                self.current_state = self._detect_page_state()
                print(f"📄 Current page state: {self.current_state.value}")
                
                # Handle the current state
                result = self._handle_current_state()
                
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
   
    def _detect_page_state_with_ai(self) -> PageState:
        """Use AI to detect page state when page detector is uncertain"""
        try:
            client = genai.Client(api_key="AIzaSyAU6PHwVlJJV5kogd4Es9hNf2Xy74fAOiA")
            
            page_content = self.page.content()
            screenshot = self.page.screenshot()
            screenshot_base64 = base64.b64encode(screenshot).decode('utf-8')
            screenshot_part = types.Part.from_bytes(
                data=screenshot_base64,
                mime_type="image/png"
            )
            prompt = f"""
            Analyze the HTML page below and classify it as exactly ONE of the following page types.
            You are also given a screenshot of the page. Use the screenshot to help you determine the page type.
            
            Each type is **mutually exclusive**. Select the most specific, primary function the page serves, based ONLY on what the user is clearly meant to do NOW. DO NOT guess—choose "unknown" if no category matches perfectly.

            Possible page types and definitions (choose ONLY one, read each CAREFULLY):

            1. intermediary_website
            - The page’s SOLE purpose is to hand off, redirect, or bridge between other sites or sections.
            - Strong indicators: There could be a modal or popup with a button to continue to the next page, or if there's no modal, there could be a button to continue to the next page.
            - **DO NOT select if ANY other user action is required** (such as login, verification, form fill, job selection).
            - There might be job listings, application forms, single job details, or confirmation messages present.

            2. login_website
            - The only main actionable area is a form to SIGN IN or authenticate.
            - Features: username/email and password fields, “Sign In”/“Log in” buttons, SSO or OAuth (Google, LinkedIn) options.
            - **User must provide credentials to proceed**.
            - DO NOT select if the user can proceed without logging in, or if the page’s main action is verification or input of non-credential information.

            3. verification_website
            - The sole main action is to complete a security or identity check, NOT login.
            - Features: CAPTCHA (“I am not a robot”), SMS/email 2FA code entry, authentication app prompt, email link confirmation.
            - User cannot proceed without passing the verification.
            - **DO NOT select if the main page function is logging in, browsing jobs, or filling any application details**.

            4. input_needed
            - The ONLY way to proceed is to provide a single, non-sensitive, non-login, non-application piece of info (NOT a full form or credential).
            - Examples: “What is your location?”, “Are you eligible to work in the UK?”, “Please enter your date of birth”, checkbox consent.
            - The mini form could be in a modal or popup and will contain only a few fields.
            - DO NOT select if the page contains or leads to an application form, login, or verification.

            5. job_application_form
            - The page’s MAIN purpose is to let the user apply for a SPECIFIC job by submitting a detailed form.
            - Features: multi-field input (name, email, phone), resume/CV/cover letter upload, multi-step process, required personal details.
            - The page should display job details (title, company) and collect info for application.
            - DO NOT select if only a single non-detailed input is required, if a modal confirmation appears, or if the application is already submitted.
            - If there is just a Apply button, this is not a job application form.

            6. application_modal_form
            - After submitting an application, a MODAL or POPUP (not a new page) appears OVER the existing content with a message like “Thank you for your application”, “Application received”, “Continue”, or next-step prompts (e.g. “create an account to save your application”).
            - The confirmation or next step is NOT the entire page—it's an overlay/modal, with the underlying page visible/dimmed.
            - DO NOT select if this message is the main page content (use “application_submitted”).

            7. application_submitted
            - The MAIN page content (not just a modal) confirms a job application was successfully submitted.
            - Features: prominent message such as “Thank you for your application”, “Application received”, “We’ll be in touch”, or clear submission confirmation.
            - **The user is NOT asked for any more information**—this is a final, stand-alone confirmation screen.
            - This could be a modal or popup with an "Applied", "Submitted", "Received", "Thank you" message.
            - DO NOT select if the confirmation is only in a modal overlay (use “application_modal_form”), or if the page indicates you have already applied (use “already_applied”).

            8. already_applied
            - The page confirms that the user has ALREADY applied for the job, and CANNOT reapply.
            - Features: explicit message like “You have already applied for this job”, “Already applied”, “Application previously submitted”, or similar.
            - The user is NOT asked for further action or information.
            - DO NOT select if this is a generic confirmation (use “application_submitted”).

            9. application_submitted_modal
            - A modal appears after submitting an application, with a message like “Thank you for your application”, “Application received”, "Applied" or “We’ll be in touch”.
            - The modal is displayed over the existing page content, and the user cannot interact with the underlying page until the modal is closed.
            - DO NOT select if this message is the main page content (use “application_submitted”).

            Instructions:
            - Analyze ALL visible main content, forms, and user actions—IGNORE nav bars, ads, or unrelated widgets.
            - If the page fits more than one category, **pick the most specific, immediate user task (not the broadest context)**.
            - Return ONLY one of these strings, and NOTHING else:

            "intermediary_website"
            "login_website"
            "verification_website"
            "input_needed"
            "job_application_form"
            "application_modal_form"
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
                    system_instruction="You are a page type classifier. Return only the page type as a string."
                )
            )
            
            result = response.text.strip().lower()
            
            # Map AI result to PageState
            state_mapping = {
                'job_board_website': PageState.JOB_BOARD_WEBSITE,
                'intermediary_website': PageState.INTERMEDIARY_WEBSITE,
                'login_website': PageState.LOGIN_WEBSITE,
                'verification_website': PageState.VERIFICATION_WEBSITE,
                'input_needed': PageState.INPUT_NEEDED,
                'job_application_form': PageState.JOB_APPLICATION_FORM,
                'application_modal_form': PageState.APPLICATION_MODAL_FORM,
                'application_submitted': PageState.APPLICATION_SUBMITTED,
                'already_applied': PageState.ALREADY_APPLIED,
                'application_submitted_modal': PageState.APPLICATION_SUBMITTED_MODAL,
                'unknown': PageState.UNKNOWN
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
    
    def _handle_current_state(self) -> NavigationResult:
        """Handle the current page state and determine next action"""
        iframe_context = self.detect_and_handle_iframes()
        if iframe_context['use_iframe_context']:
            frame = iframe_context['iframe_context']['frame']
        else:
            frame = None
        
        try:
            if self.current_state == PageState.JOB_BOARD_WEBSITE:
                return self._handle_job_board_website()
            elif self.current_state == PageState.INTERMEDIARY_WEBSITE:
                return self._handle_intermediary_website()
            elif self.current_state == PageState.LOGIN_WEBSITE:
                return self._handle_login_website()
            elif self.current_state == PageState.VERIFICATION_WEBSITE:
                return self._handle_verification_website()
            elif self.current_state == PageState.INPUT_NEEDED or self.current_state == PageState.APPLICATION_MODAL_FORM:
                return self._handle_input_needed(self.preferences, frame)
            elif self.current_state == PageState.JOB_APPLICATION_FORM:
                return self._handle_job_application_form()
            elif self.current_state == PageState.APPLICATION_SUBMITTED or self.current_state == PageState.APPLICATION_SUBMITTED_MODAL:
                return NavigationResult(
                    success=True,
                    try_restart=False,
                    form_ready=False,
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
    
    def _convert_text_to_actions(self, page_content: str) -> str:
        try:
            client = genai.Client(api_key="AIzaSyAU6PHwVlJJV5kogd4Es9hNf2Xy74fAOiA")
            
            prompt = f"""
                You are given the HTML content of a webpage.
                Your task is to **analyze the page and describe in a single, clear paragraph the specific user actions required to progress to the next page** in a typical web flow.

                Actions to describe (ONLY if they are clearly present and visible on the page):
                1. **Click** - Any visible, interactive element that must be clicked to proceed (buttons, links, icons, clickable divs, elements with role="button", etc.).
                2. **Fill input** – Any visible, enabled input field that must be filled, checked, or selected before proceeding. This includes:
                - Text fields: Describe what value should be typed, using the input’s label, placeholder, or aria-label (e.g., "type your email address in the 'Email' field").
                - Radio buttons and checkboxes: State which one to select or check, always using the visible label, group question, or their location for disambiguation.
                - Dropdowns (native or custom): Instruct the user to "select the best option from the dropdown," always referencing it by its visible label, placeholder, or clear location (e.g., "select the best option from the 'Country' dropdown at the top of the form").
                - File upload: If a visible upload button is present (not just an input), instruct the user to click the upload button (describe it by label, aria-label, or icon and location) and select their file.

                **Critical Guidelines:**
                - Only describe actions for fields that are **actually visible and interactive** (do NOT include hidden, disabled, off-screen, or future-step elements).
                - **Never invent or assume actions that are not strictly evident from the HTML.**
                - Always use visible text, labels, aria-labels, or placeholders to refer to elements, and specify their location if there could be confusion (e.g., "at the bottom right", "just below the main heading", "in the center of the form").
                - For dropdowns, never state which option to select; always instruct to "select the best option".
                - Do not refer to elements by CSS class, color, or purely visual cues (like "the blue button").
                - Do not output code, lists, or JSON—**return only a clear, complete, descriptive paragraph.**

                ---

                **Good Example 1:**  
                "To continue, type your email address in the 'Email' input field at the top of the form, select the best option from the 'Country' dropdown just below the email field, check the box labeled 'Subscribe to newsletter' at the bottom left, and then click the 'Next' button in the bottom right corner of the page."

                *Why good?*  
                - Identifies each input by label and location
                - Dropdown is referenced with label and position, and the instruction is to select the best option

                **Good Example 2:**  
                "Begin by selecting the 'Yes' radio button beneath the eligibility question, then select the best option from the dropdown labeled 'Department' in the center of the page, and finally, click the circular icon with the arrow at the bottom right to proceed."

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

                ---

                HTML content:
                {page_content}
                """
            
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction="You are an action determiner. Return only the action type as a string.",
                    response_mime_type="text/plain"
                )
            )
            
            actions_paragraph = response.text.strip()
            
            return actions_paragraph
                    
        except Exception as e:
            print(f"❌ Error generating content: {e}")
            return ""
    
    def _convert_actions_to_interactions(self, text: str, preferences: Dict[str, Any], page_content: str) -> PageInteractionResponse:
        """Convert a text description of actions to a list of interactions"""
        try:
            client = genai.Client(api_key="AIzaSyAU6PHwVlJJV5kogd4Es9hNf2Xy74fAOiA")
            
            prompt = f"""
            You are given a paragraph describing the actions that need to be taken to progress to the next page, a dictionary of user preferences, and the HTML content of the page.

            Your task is to convert the paragraph into a structured list of interactions, using these rules and best practices:

            STRUCTURE
            ---------
            For each action, return a JSON object matching this schema (PageInteraction):
            - interaction_type: "click", "fill_input", or "type" (see details below)
            - element_selector: Robust, minimal CSS selector for the interactive element
            - input_type: "text", "date", "file", "radio", "checkbox", "dropdown", or "div"
            - expected_value: The value to use (from preferences if relevant), or null if not applicable

            USER PREFERENCES
            ----------------
            A Python dictionary of user preferences will be provided. Use these for `expected_value` if the step is about filling a known value (e.g., first name, email, date of birth, country, etc).

            SELECTOR GUIDELINES
            -------------------
            1. Only return selectors for interactive, user-visible, enabled, non-hidden fields and buttons.
            2. Never use iframe selectors or cross-document selectors. If an element is inside an iframe, build the selector as if already inside the iframe's DOM.
            3. Use robust selectors based on labels, aria-labels, placeholders, type, name, id, and unique classes (avoid nth-child, generic divs, or positional selectors unless no other stable choice exists).
            4. Example of GOOD selectors:
                input[name='first_name']
                input[type='email']
                input[label='Email Address']
                select[name='country']
                div.select__control
                button:has-text('Continue')
                input[type='checkbox'][label='I agree']
                button[aria-label='Upload resume']
                div.upload-btn[role='button']
            5. Example of BAD selectors:
                iframe#grnhse_iframe input[name='first_name']
                div:nth-child(4)
                body button
                input[type='file']   # for upload, do not use input!
                div                  # hopelessly generic
                .modal[style*='display:none'] button
                button:nth-child(3)
                label                # label, not input
                span                 # span, not interactive field

            ACTION MAPPING
            --------------
            - Typing or filling a value: Use "type" or "fill_input" as interaction_type; input_type should be "text", "date", "file", etc.
            - Selecting an option: For native selects or custom dropdowns, use input_type "dropdown", expected_value = value from preferences if clearly specified, otherwise null.
            - Radio/checkbox: Use "fill_input" with input_type "radio" or "checkbox", expected_value should be the label or value to select.
            - Buttons, clickable divs/icons: Use "click", input_type "div" (or "file" if it opens a file chooser), expected_value is always null.

            VISIBILITY & INTERACTIVITY
            --------------------------
            - Only include fields/buttons that are currently visible and interactive (NOT hidden by CSS, off-screen, disabled, or future steps).
            - Never include elements not visible to the user.

            FIELD-SPECIFIC NOTES
            --------------------
            - Text fields: Must not be dropdowns, checkboxes, or radios disguised as inputs.
            - Dropdown/select: Target native <select> or visible custom dropdown controls (like div.select__control).
            - Radio: Only include visible, interactive radios; value should match preferences if provided and contextually relevant.
            - Checkbox: Only include if visible and checked/unchecked as required by preferences/context.
            - Upload button: Only return the selector for a <button>, <a>, or <div> acting as a file upload trigger (never <input>).
            - Never use selectors that include iframe, or that are not stable across reloads.

            OUTPUT
            ------
            Return a JSON array (list) of PageInteraction objects, one for each required action.

            EXAMPLES
            --------

            GOOD Example 1:
            Paragraph:
            "Type your first name in the input labeled 'First Name', select the best option from the 'Country' dropdown, and click the 'Continue' button at the bottom right."
            Preferences: {{ "first_name": "Sam", "country": "United Kingdom" }}
            Output:
            [
                {{
                    "interaction_type": "type",
                    "element_selector": "input[label='First Name']",
                    "input_type": "text",
                    "expected_value": "Sam"
                }},
                {{
                    "interaction_type": "fill_input",
                    "element_selector": "select[label='Country']",
                    "input_type": "dropdown",
                    "expected_value": null
                }},
                {{
                    "interaction_type": "click",
                    "element_selector": "button:has-text('Continue')",
                    "input_type": "div",
                    "expected_value": null
                }}
            ]

            GOOD Example 2:
            Paragraph:
            "Enter your email in the input labeled 'Email Address', check the box labeled 'I agree to the terms', and then click the icon at the bottom right to continue."
            Preferences: {{ "email": "sam@email.com" }}
            Output:
            [
                {{
                    "interaction_type": "type",
                    "element_selector": "input[label='Email Address']",
                    "input_type": "text",
                    "expected_value": "sam@email.com"
                }},
                {{
                    "interaction_type": "fill_input",
                    "element_selector": "input[type='checkbox'][label='I agree to the terms']",
                    "input_type": "checkbox",
                    "expected_value": "I agree to the terms"
                }},
                {{
                    "interaction_type": "click",
                    "element_selector": "div[role='button'][aria-label='Continue']",
                    "input_type": "div",
                    "expected_value": null
                }}
            ]

            GOOD Example 3:
            Paragraph:
            "Type your date of birth in the 'Date of Birth' field near the top, and then select the best option from the 'Department' dropdown below."
            Preferences: {{ "date_of_birth": "1990-01-01", "department": "Engineering" }}
            Output:
            [
                {{
                    "interaction_type": "type",
                    "element_selector": "input[label='Date of Birth']",
                    "input_type": "date",
                    "expected_value": "1990-01-01"
                }},
                {{
                    "interaction_type": "fill_input",
                    "element_selector": "select[label='Department']",
                    "input_type": "dropdown",
                    "expected_value": null
                }}
            ]

            GOOD Example 4:
            Paragraph:
            "Check the box labeled 'Subscribe to updates' above the submit button, and then click 'Submit'."
            Preferences: {{ "subscribe": true }}
            Output:
            [
                {{
                    "interaction_type": "fill_input",
                    "element_selector": "input[type='checkbox'][label='Subscribe to updates']",
                    "input_type": "checkbox",
                    "expected_value": "Subscribe to updates"
                }},
                {{
                    "interaction_type": "click",
                    "element_selector": "button:has-text('Submit')",
                    "input_type": "div",
                    "expected_value": null
                }}
            ]

            BAD Example 1:
            [
                {{
                    "interaction_type": "type",
                    "element_selector": "input[label='Email Address']",
                    "input_type": "text",
                    "expected_value": null
                }}
            ]
            Why bad: Expected value should be from preferences.

            BAD Example 2:
            [
                {{
                    "interaction_type": "fill_input",
                    "element_selector": "select[label='Department']",
                    "input_type": "dropdown",
                    "expected_value": "Engineering"
                }}
            ]
            Why bad: Paragraph does not specify the value.

            BAD Example 3:
            [
                {{
                    "interaction_type": "click",
                    "element_selector": "div:nth-child(4)",
                    "input_type": "div",
                    "expected_value": null
                }}
            ]
            Why bad: Non-robust, positional selector.

            BAD Example 4:
            [
                {{
                    "interaction_type": "fill_input",
                    "element_selector": "input[type='checkbox'][label='I agree']",
                    "input_type": "checkbox"
                }}
            ]
            Why bad: Omits expected_value.

            ---

            ALWAYS output ONLY the JSON list of PageInteraction objects, nothing else.

            ---

            Paragraph:
            {text}

            Preferences:
            {json.dumps(preferences)}

            HTML content:
            {page_content}
            """
            
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction="You are an action converter. Return only the interactions as a list of JSON objects.",
                    response_mime_type="application/json",
                    response_schema=PageInteractionResponse
                )
            )
            
            response = response.parsed
            
            return response
        
        except Exception as e:
            print(f"❌ Error converting actions: {e}")
            return PageInteractionResponse(interactions=[])
    
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
            
            if not self._is_element_interactive(element, frame):
                context_str = "iframe" if frame else "page"
                print(f"⏭️ Skipping non-interactive element in {context_str}: {interaction.element_selector}")
                return True
            
            if not self._is_element_in_form_context(element, frame):
                context_str = "iframe" if frame else "page"
                print(f"⏭️ Skipping element outside form context in {context_str}: {interaction.element_selector}")
                return True
            
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
            
            if not self._is_element_interactive(element, frame):
                context_str = "iframe" if frame else "page"
                print(f"⏭️ Skipping non-interactive radio button in {context_str}: {interaction.element_selector}")
                return True
            
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
            
            if not self._is_element_interactive(element, frame):
                context_str = "iframe" if frame else "page"
                print(f"⏭️ Skipping non-interactive element in {context_str}: {interaction.element_selector}")
                return True
            
            if not self._is_element_in_form_context(element, frame):
                context_str = "iframe" if frame else "page"
                print(f"⏭️ Skipping element outside form context in {context_str}: {interaction.element_selector}")
                return True
            
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
    
    def _handle_input_needed(self, preferences: Dict[str, Any], frame=None) -> NavigationResult:
        """Handle pages requiring specific input - get all inputs and fill them"""
        try:
            print("📝 Input needed page detected - getting all inputs...")
            
            # Figure out what needs to be done in the page
            actions_paragraph = self._convert_text_to_actions(self.page.content())
            interactions = self._convert_actions_to_interactions(actions_paragraph, preferences, self.page.content())
            
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
                form_ready=True
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