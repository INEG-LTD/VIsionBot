#!/usr/bin/env python3
"""
Application Filler

A comprehensive job application form filler that can handle various form types
and automatically fill them based on user preferences using gpt-5-mini analysis.
"""

# Removed incorrect import
import time
import json
from typing import Callable, List, Dict, Optional, Any, Tuple
from playwright.sync_api import Page
from pydantic import BaseModel
from enum import Enum
import traceback

from urllib.parse import urlparse
from yaspin import Spinner, yaspin
from yaspin.api import Yaspin
from yaspin.spinners import Spinners

from ai_utils import generate_model, generate_text
from terminal import term

from bot_utils import debug_mode, start_browser, dprint

# Import removed - using self.preferences instead

class FieldType(Enum):
    TEXT = "text"
    SELECT = "select"
    RADIO = "radio"
    CHECKBOX = "checkbox"
    UPLOAD = "upload"
    SUBMIT = "submit"

# Base class for all application fields
class BaseApplicationField(BaseModel):
    field_order: int
    field_name: str
    field_selector: str
    field_is_visible: bool
    field_in_form: bool
    field_required: Optional[bool] = None
    field_placeholder: Optional[str] = None

# Text input field
class TextApplicationField(BaseApplicationField):
    field_type: FieldType = FieldType.TEXT
    field_value: Optional[str] = None
    field_max_length: Optional[int] = None
    field_pattern: Optional[str] = None
    
class TextApplicationFieldResponse(BaseModel):
    fields: List[TextApplicationField]

# Select dropdown field
class SelectApplicationField(BaseApplicationField):
    field_type: FieldType = FieldType.SELECT
    field_value: Optional[str] = None
    field_options: Optional[List[str]] = None
    field_multiple: Optional[bool] = False

class SelectApplicationFieldResponse(BaseModel):
    fields: List[SelectApplicationField] = []

# Radio button field
class RadioApplicationField(BaseApplicationField):
    field_type: FieldType = FieldType.RADIO
    field_value: Optional[bool] = None
    field_options: Optional[List[str]] = None
    field_group_name: Optional[str] = None

class RadioApplicationFieldResponse(BaseModel):
    fields: List[RadioApplicationField]

# Checkbox field
class CheckboxApplicationField(BaseApplicationField):
    field_type: FieldType = FieldType.CHECKBOX
    field_value: Optional[bool] = True
    field_checked: Optional[bool] = False

class CheckboxApplicationFieldResponse(BaseModel):
    fields: List[CheckboxApplicationField]

# File upload field
class UploadApplicationField(BaseApplicationField):
    field_type: FieldType = FieldType.UPLOAD
    field_value: Optional[bool] = None
    field_file_path: Optional[str] = None
    field_accepted_types: Optional[List[str]] = None
    field_max_size: Optional[int] = None

class SubmitButtonApplicationField(BaseApplicationField):
    field_type: FieldType = FieldType.SUBMIT
    field_selector: Optional[str] = None
    field_is_visible: Optional[bool] = None
    field_in_form: Optional[bool] = None
    field_required: Optional[bool] = None
    
class ApplicationStateResponse(BaseModel):
    submitted: bool
    completed: bool
    error_in_submission: bool
    verification_required: bool
    more_forms: bool
    reason: str

class ApplicationFiller:
    def __init__(self, page: Page = None, preferences: Dict[str, Any] = None):
        """
        Initialize the Application Filler
        
        Args:
            page: Playwright page object
            preferences: User preferences for filling forms
        """
        if page is None:
            _, _, new_page = start_browser()
            self.page = new_page
        else:
            self.page = page
            
        self.preferences = preferences or {}
        self.max_form_iterations = 10  # Prevent infinite loops
        self.current_iteration = 0
        self.old_url = None
    
    def _serialize_field_for_json(self, field) -> dict:
        """
        Custom serialization function to handle FieldType enums properly
        """
        field_dict = field.model_dump()
        # Convert FieldType enum to string for JSON serialization
        if 'field_type' in field_dict and isinstance(field_dict['field_type'], FieldType):
            field_dict['field_type'] = field_dict['field_type'].value
        return field_dict
    
    def fill_application(self, on_success_callback: Callable[[], None] = None, on_failure_callback: Callable[[], None] = None) -> bool:
        """
        Main entry point for application filling process
        
        Returns:
            bool: True if application was successfully filled, False otherwise
        """
        print("🚀 Starting application filling process...")
        
        try:
            # if apply_button_result['action'] == 'restart':
            #     print("🔄 New page detected after apply button click - restarting algorithm")
            #     return self.fill_application()  # Recursive call for new page
            # elif apply_button_result['action'] == 'continue':
            #     print("✅ Apply button handled - continuing to main algorithm")
            
            # Step 2: Run main form filling algorithm
            success = self.run_main_algorithm()
            if success:
                if on_success_callback:
                    on_success_callback()
            else:
                if on_failure_callback:
                    on_failure_callback()
            
        except Exception as e:
            print(f"❌ Error in application filling: {e}")
            return False
    
    def verify_cookie_button(self, selector_to_verify: str, best_of: int = 3) -> str:
        """
        Verify the cookie policy is accepted using best_of parameter for consensus
        """
        print(f"🔍 Verifying cookie button: {selector_to_verify} (best_of: {best_of})")
        
        if best_of <= 1:
            # Single run - use original logic
            try:
                system_prompt = f"""
                    ROLE
                    You are a senior UI/DOM analyst. Verify a proposed Playwright CSS selector for the site's cookie-consent "accept" button. If it is fully valid, return it UNCHANGED. If it fails any check, IGNORE it and RECOMPUTE the correct selector from the HTML. Output EXACTLY ONE line: the selector string or an empty string.

                    INPUTS
                    - proposed_selector: {selector_to_verify}
                    - NOTE: {"The proposed selector is empty - you MUST find a cookie accept button from scratch" if not selector_to_verify else "The proposed selector exists and should be validated"}

                    TARGET (WHAT COUNTS)
                    A single, visible, enabled control in the TOP DOCUMENT (no iframes) that accepts cookies:
                    - Element is a <button>, <a>, or any element with button-like semantics (e.g., role="button").
                    - Accessible name/label implies acceptance: accept, accept all, allow, allow all, agree, i agree, ok, yes (case-insensitive).
                    - Prefer the primary "accept all" action over partial/necessary-only acceptance.

                    FORBIDDEN / EXCLUDE
                    - Anything inside an <iframe> (selector must not include iframe nodes; top document only).
                    - Hidden/disabled/off-screen/zero-size elements (display:none, visibility:hidden, opacity:0, [hidden], [aria-hidden="true"], [disabled], aria-disabled="true"]).
                    - Non-accept actions: reject/decline/deny/disagree/manage/settings/preferences/customize/options/only necessary|essential.
                    - Brittle selectors: :nth-*, inline style predicates, state-dependent hacks ([aria-expanded], transient classes).
                    - Overly generic roots: html body button, body a.

                    PREFERENCES FOR SELECTOR STABILITY
                    - Prefer stable attributes: #id, [data-*], [name], [aria-label], [title], [data-testid], [data-qa].
                    - Otherwise: select the nearest stable consent container (id/class contains cookie|consent|gdpr|privacy) + a button with :has-text(...) or attribute match.
                    - Vendor-known targets (when present; highest priority):
                    #onetrust-accept-btn-handler
                    #CybotCookiebotDialogBodyButtonAccept
                    .didomi-accept-button
                    #sp-accept, #sp-accept-all
                    #qc-cmp2-ui .qc-cmp2-accept
                    #truste-consent-button
                    #cookie_action_close_header

                    VALIDATION CHECKLIST (APPLY TO proposed_selector)
                    1) Resolves to EXACTLY ONE element in the top document.
                    2) Element is <button> or <a> or has role="button" (button-like), NOT an <input>.
                    3) Element is visible & enabled now.
                    4) Accessible name/label/text matches acceptance (positive) and does NOT match any negative/management wording.
                    5) Selector has NO iframe prefixes/nodes and is reasonably stable (uses id/data/aria/container + :has-text rather than positional/state hacks).

                    DECISION RULE
                    - If ALL checklist items pass AND no more primary "accept all" button exists in the same banner, RETURN proposed_selector UNCHANGED.
                    - Otherwise, RECOMPUTE:
                    a) Locate the active cookie/consent banner container.
                    b) Enumerate visible, enabled <button>/<a>/<div[role='button']> within it.
                    c) Filter by positive acceptance semantics; drop negatives/management/partial-only if a true "accept all" exists.
                    d) Prefer vendor-known targets; else build the most stable unique selector (id/data/aria/container + :has-text(/accept|allow|agree|ok|yes|accept all/i)).
                    e) Ensure uniqueness, visibility, top-document, and not <input>.

                    OUTPUT
                    - Output EXACTLY ONE line with ONLY the selector string (no quotes, no extra text).
                    - If no qualifying button in the top document, output an empty string.

                    REASONING PROTOCOL (DO NOT REVEAL)
                    - Validate proposed_selector with the checklist.
                    - If invalid/ambiguous or a better primary accept-all exists, recompute per procedure.
                    - Final self-check before output: uniqueness, visibility, semantics, stability, no forbidden patterns.
                    """

                accept_cookies_selector = generate_text(
                    "You are an expert at analyzing HTML and finding cookie accept buttons",
                    system_prompt=system_prompt,
                    model="gpt-5-mini"
                )
                
                print(f"🔍 Verified accept cookies button: {accept_cookies_selector}")
                return accept_cookies_selector
            except Exception as e:
                print(f"❌ Error in verify_cookie_policy: {e}")
                return selector_to_verify
        
        # Multiple runs for consensus
        results = []
        for i in range(best_of):
            try:
                print(f"🔍 Run {i+1}/{best_of} for cookie button verification...")
                
                system_prompt = f"""
                    ROLE
                    You are a senior UI/DOM analyst. Verify a proposed Playwright CSS selector for the site's cookie-consent "accept" button. If it is fully valid, return it UNCHANGED. If it fails any check, IGNORE it and RECOMPUTE the correct selector from the HTML. Output EXACTLY ONE line: the selector string or an empty string.

                    INPUTS
                    - proposed_selector: {selector_to_verify}
                    - NOTE: {"The proposed selector is empty - you MUST find a cookie accept button from scratch" if not selector_to_verify else "The proposed selector exists and should be validated"}

                    TARGET (WHAT COUNTS)
                    A single, visible, enabled control in the TOP DOCUMENT (no iframes) that accepts cookies:
                    - Element is a <button>, <a>, or any element with button-like semantics (e.g., role="button").
                    - Accessible name/label implies acceptance: accept, accept all, allow, allow all, agree, i agree, ok, yes (case-insensitive).
                    - Prefer the primary "accept all" action over partial/necessary-only acceptance.

                    FORBIDDEN / EXCLUDE
                    - Anything inside an <iframe> (selector must not include iframe nodes; top document only).
                    - Hidden/disabled/off-screen/zero-size elements (display:none, visibility:hidden, opacity:0, [hidden], [aria-hidden="true"], [disabled], aria-disabled="true"]).
                    - Non-accept actions: reject/decline/deny/disagree/manage/settings/preferences/customize/options/only necessary|essential.
                    - Brittle selectors: :nth-*, inline style predicates, state-dependent hacks
                    - Overly generic roots: html body button, body a.

                    PREFERENCES FOR SELECTOR STABILITY
                    - Prefer stable attributes: #id, [data-*], [name], [aria-label], [title], [data-testid], [data-qa].
                    - Otherwise: select the nearest stable consent container (id/class contains cookie|consent|gdpr|privacy) + a button with :has-text(...) or attribute match.
                    - Vendor-known targets (when present; highest priority):
                    #onetrust-accept-btn-handler
                    #CybotCookiebotDialogBodyButtonAccept
                    .didomi-accept-button
                    #sp-accept, #sp-accept-all
                    #qc-cmp2-ui .qc-cmp2-accept
                    #truste-consent-button
                    #cookie_action_close_header

                    VALIDATION CHECKLIST (APPLY TO proposed_selector)
                    1) Resolves to EXACTLY ONE element in the top document.
                    2) Element is <button> or <a> or has role="button" (button-like), NOT an <input>.
                    3) Element is visible & enabled now.
                    4) Accessible name/label/text matches acceptance (positive) and does NOT match any negative/management wording.
                    5) Selector has NO iframe prefixes/nodes and is reasonably stable (uses id/data/aria/container + :has-text rather than positional/state hacks).

                    DECISION RULE
                    - If ALL checklist items pass AND no more primary "accept all" button exists in the same banner, RETURN proposed_selector UNCHANGED.
                    - Otherwise, RECOMPUTE:
                    a) Locate the active cookie/consent banner container.
                    b) Enumerate visible, enabled <button>/<a>/<div[role='button']> within it.
                    c) Filter by positive acceptance semantics; drop negatives/management/partial-only if a true "accept all" exists.
                    d) Prefer vendor-known targets; else build the most stable unique selector (id/data/aria/container + :has-text(/accept|allow|agree|ok|yes|accept all/i)).
                    e) Ensure uniqueness, visibility, top-document, and not <input>.

                    OUTPUT
                    - Output EXACTLY ONE line with ONLY the selector string (no quotes, no extra text).
                    - If no qualifying button in the top document, output an empty string.

                    REASONING PROTOCOL (DO NOT REVEAL)
                    - Validate proposed_selector with the checklist.
                    - If fnvalid/ambiguous or a better primary accept-all exists, recompute per procedure.
                    - Final self-check before output: uniqueness, visibility, semantics, stability, no forbidden patterns.
                    """

                result = generate_text(
                    "You are an expert at analyzing HTML and finding cookie accept buttons",
                    system_prompt=system_prompt,
                    model="gpt-5-mini"
                )
                
                if result and result.strip():
                    results.append(result.strip())
                    print(f"🔍 Run {i+1} result: {result.strip()}")
                else:
                    print(f"🔍 Run {i+1} returned empty result")
                    
            except Exception as e:
                print(f"❌ Error in run {i+1}: {e}")
                continue
        
        if not results:
            print(f"❌ All {best_of} runs failed, returning original selector")
            return selector_to_verify
        
        # Find most common result
        from collections import Counter
        result_counts = Counter(results)
        most_common_result = result_counts.most_common(1)[0]
        
        print(f"🔍 Consensus results: {dict(result_counts)}")
        print(f"🔍 Most common result: {most_common_result[0]} (frequency: {most_common_result[1]}/{best_of})")
        
        # If most common result appears more than once and is different from original, use it
        if most_common_result[1] > 1 and most_common_result[0] != selector_to_verify:
            print(f"🔍 Using consensus result: {most_common_result[0]}")
            return most_common_result[0]
        else:
            print(f"🔍 No clear consensus, returning last result: {results[-1] if results else selector_to_verify}")
            return results[-1] if results else selector_to_verify
    
    def click_accept_cookies_button(self) -> bool:
        """
        Click the accept cookies button
        
        Returns:
            bool: True if button was found and clicked successfully
        """
        try:
            if self.old_url is None:
                self.old_url = self.page.url
            else:
                # Check the domain of the old url and the new url
                old_domain = urlparse(self.old_url).netloc
                new_domain = urlparse(self.page.url).netloc
                if old_domain == new_domain:
                    return True
            
            # Common accept cookies button selectors
            with yaspin(text="Looking for accept cookies button...", color="cyan") as spinner:
                try:
                    system_prompt = """
                    ROLE
                    You are a senior UI/DOM analyst. Return a single Playwright-compatible selector for the site's cookie-consent "accept" button.

                    INPUT
                    - html: full page HTML

                    REASONING PROTOCOL (DO NOT REVEAL)
                    1) Locate the active cookie/consent banner container.
                    2) Enumerate candidate buttons whose accessible name implies acceptance: accept, accept all, allow, allow all, agree, i agree, ok, yes (case-insensitive).
                    3) Exclude negatives: reject, decline, deny, disagree, manage, settings, preferences, customize, options, only necessary/essential.
                    4) Keep only visible, enabled, top-document elements (ignore iframes, hidden/disabled/off-screen).
                    5) Prefer the primary "accept all" action over partial/necessary-only.
                    6) Build a unique, stable selector. Prefer vendor IDs first; otherwise scope to the consent container + a text match.
                    7) Self-check: selector resolves to exactly one element; no positional or state-dependent hacks.

                    RULES
                    - Prefer stable attributes: id, data-*, name, aria-label, title, data-testid.
                    - Known vendor targets (when present): 
                    #onetrust-accept-btn-handler
                    #CybotCookiebotDialogBodyButtonAccept
                    .didomi-accept-button
                    #sp-accept, #sp-accept-all
                    #qc-cmp2-ui .qc-cmp2-accept
                    #truste-consent-button
                    #cookie_action_close_header
                    - Otherwise: pick nearest stable consent container (id/class contains cookie|consent|gdpr|privacy) then a button with :has-text(...) or attribute match.
                    - Avoid brittle selectors: :nth-*, inline style filters, transient state ([aria-expanded], [style*='display:none'], [aria-hidden]).
                    - Do NOT return anything inside an <iframe>; frame scoping is out of scope for a single selector string.

                    OUTPUT
                    - Output EXACTLY one line containing ONLY the selector string (no quotes, no extra text).
                    - If no qualifying button in the top document, output an empty string.

                    GOOD EXAMPLES (DO NOT OUTPUT)
                    #onetrust-accept-btn-handler
                    button#CybotCookiebotDialogBodyButtonAccept
                    button.didomi-accept-button
                    div#cookie-banner button:has-text("Accept all")
                    section.consent-modal button:has-text(/accept|allow|agree/i)

                    BAD EXAMPLES (DO NOT OUTPUT)
                    button:nth-of-type(2)                 # positional
                    [aria-hidden="true"] button           # hidden
                    iframe[name="sp_message_iframe"] ...  # requires iframe handling
                    a.manage-preferences                  # wrong action
                    """

                    
                    prompt = f"""
                    Here is the HTML of the page:
                    "{self.page.content()}"
                    """
                    
                    accept_cookies_selector = generate_text(prompt, system_prompt=system_prompt, model="gpt-5-mini")
                    
                    verified_accept_cookies_selector = self.verify_cookie_button(accept_cookies_selector)
                    accept_cookies_selector = verified_accept_cookies_selector
                    
                    try:
                        accept_cookies_button = None
                        
                        accept_cookies_button = self.page.locator(accept_cookies_selector)
                        if accept_cookies_button and accept_cookies_button.is_visible():
                            if debug_mode:
                                spinner.write("Found accept cookies button, clicking...")

                            # Click the accept cookies button
                            accept_cookies_button.click()
                            spinner.hide()
                            print(term.green + "> Accepted cookies on " + term.normal + self.page.url)
                        else:
                            dprint(f"❌ No accept cookies button found with selector: {accept_cookies_selector}")
                            spinner.hide()
                            return False
                    except Exception as e:
                        dprint(f"❌ Error finding accept cookies button: {e}")
                        spinner.hide()
                        return False
                    
                    return True
                except Exception as e:
                    spinner.hide()
                    print(f"❌ Error clicking accept cookies button: {e}")
                    return False
            
        except Exception as e:
            print(f"❌ Error clicking accept cookies button: {e}")
            return False
    
    def run_main_algorithm(self) -> bool:
        """
        Main form filling algorithm that continues until completion
        
        Returns:
            bool: True if all forms were successfully filled
        """
        print(term.bold + "🎯 Starting Application Filler..." + term.reset)
        
        self.current_iteration = 0
        
        while self.current_iteration < self.max_form_iterations:
            self.current_iteration += 1
            dprint(f"\n🔄 Form iteration {self.current_iteration}/{self.max_form_iterations}")
            
            
            try:
                # Step 0: Click accept cookies button
                self.click_accept_cookies_button()
                
                self.old_url = self.page.url
                
                # Step 1: Detect iframes and determine context
                iframe_context = self.detect_and_handle_iframes()

                # Step 2: Find all form fields based on context
                with yaspin(Spinners.binary, text="Finding all form fields...", color="cyan") as spinner:
                    if iframe_context['use_iframe_context']:
                        # Use unified field detection functions with frame context
                        frame = iframe_context['iframe_context']['frame']
                        text_fields, select_fields, radio_fields, checkbox_fields, upload_fields = self.find_all_form_inputs(frame, iframe_context["iframe_context"], spinner)
                    else:
                        # Use unified field detection functions for main page
                        text_fields, select_fields, radio_fields, checkbox_fields, upload_fields = self.find_all_form_inputs(spinner=spinner)
                    
                    spinner.hide()
                
                if not text_fields and not select_fields and not radio_fields and not checkbox_fields and not upload_fields:
                    print("Hmmmm, this is weird. We couldn't find any job application form fields on this page.")
                    print("Please check if the page is a job application form.")
                    print("If it is, please report this as a bug.")
                    return False
                
                initial_screenshot_bytes, initial_screenshot_context = self._take_smart_screenshot(frame, iframe_context)
                
                total_fields = (len(text_fields) + len(select_fields) + 
                               len(radio_fields) + len(checkbox_fields) + 
                               len(upload_fields))
                
                print(f"📋 Found {total_fields} total inputs:")
                print(f"  - {len(text_fields)} text input fields")
                print(f"  - {len(radio_fields)} radio button groups")
                print(f"  - {len(checkbox_fields)} checkboxes")
                print(f"  - {len(select_fields)} select fields")
                print(f"  - {len(upload_fields)} upload buttons")
                
                # Print field details
                for field in text_fields:
                    print(f"Text Field: {field.field_name} - {field.field_value} - {field.field_selector}")
                for field in select_fields:
                    print(f"Select Field: {field.field_name} - {field.field_value} ({field.field_options}) - {field.field_selector}")
                for field in radio_fields:
                    print(f"Radio Field: {field.field_name} - {field.field_value} ({field.field_options}) - {field.field_selector}")
                for field in checkbox_fields:
                    print(f"Checkbox Field: {field.field_name} - {field.field_value} - {field.field_selector}")
                for field in upload_fields:
                    print(f"Upload Field: {field.field_name} - {field.field_value} - {field.field_selector}")
                    
                # Step 3: Fill all form fields
                frame = iframe_context['iframe_context']['frame'] if iframe_context['use_iframe_context'] else None
                found_fields = self.fill_all_form_inputs(text_fields, select_fields, radio_fields, checkbox_fields, upload_fields, frame, iframe_context)
                if not found_fields:
                    print("❌ Failed to fill form inputs")
                    continue
                
                # Step 4: Find a next button if there is one
                next_button = self.find_and_click_next_button()
                if not next_button:
                    print("❌ Failed to find or click next button")
                    print("🔍 Checking for submit button")
                else:
                    print("✅ Next button found and clicked")
                    state_result = self.check_form_submission_with_gpt(initial_screenshot_bytes, frame, iframe_context)
                    if state_result.error_in_submission:
                        print("⚠️ Error in submission - continuing...")
                        self.page.pause()
                        continue
                    elif state_result.verification_required:
                        print("⚠️ Verification required - continuing...")
                        self.page.pause()
                        continue
                    elif state_result.more_forms:
                        print("➡️ More forms detected - continuing...")
                        self.page.pause()
                        continue
                    elif state_result.submitted and state_result.completed:
                        print("✅ Form successfully submitted!")
                        return True
                    else:
                        print("⚠️ Form submission may have failed or requires attention")
                        # Continue to next iteration to handle any new fields
                        continue
                            
                # Step 5: Find and click submit button
                submit_btn = self.find_submit_button_with_gpt(frame)
                if not submit_btn:
                    print("❌ Failed to find or click submit input")
                    return False
                else:
                    if submit_btn.field_selector is None:
                        print("❌ No submit button found")
                        return False
                        
                    print(f"🔍 Clicking submit button with selector: {submit_btn.field_selector}")
                    if frame:
                        element = frame.locator(submit_btn.field_selector)
                    else:
                        element = self.page.locator(submit_btn.field_selector)
                    
                    if not element:
                        print(f"⚠️ Submit button not found with selector: {submit_btn.field_selector}")
                        return False
                    
                    element.click()
                
                # Step 6: Analyze if form was submitted successfully
                submission_result = self.check_form_submission_with_gpt(initial_screenshot_bytes, frame, iframe_context)
                
                if submission_result.submitted and submission_result.completed:
                    print("✅ Form successfully submitted!")
                    print("🎉 Application process completed!")
                    return True
                elif submission_result.error_in_submission:
                    print("⚠️ Error in submission - continuing...")
                    self.page.pause()
                    continue
                elif submission_result.verification_required:
                    print("⚠️ Verification required - continuing...")
                    self.page.pause()
                    continue
                elif submission_result.more_forms:
                    print("➡️ More forms detected - continuing...")
                    self.page.pause()
                    continue
                else:
                    print("⚠️ Form submission may have failed or requires attention")
                    # Continue to next iteration to handle any new fields
                    continue
                    
            except Exception as e:
                print(f"❌ Error in main algorithm iteration {self.current_iteration}: {e}")
                traceback.print_exc()
                continue
        
        print(f"⚠️ Reached maximum iterations ({self.max_form_iterations})")
        return False
        
    def detect_and_handle_iframes(self) -> Dict[str, Any]:
        """
        Detect iframes on the page and determine if form fields are inside them
        
        Returns:
            Dict with iframe information and whether to use iframe context
        """
        try:
            dprint("🔍 Detecting iframes on the page...")
            
            # Find all iframes on the page
            iframes = self.page.query_selector_all('iframe')
            visible_iframes = [iframe for iframe in iframes if iframe.is_visible()]
            
            dprint(f"📋 Found {len(iframes)} total iframes, {len(visible_iframes)} visible")
            
            if not visible_iframes:
                dprint("ℹ️ No visible iframes found - using main page context")
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
                        dprint(f"✅ Found iframe {i+1} with {len(form_elements)} form elements")
                        iframe_with_forms = {
                            'index': i,
                            'iframe': iframe,
                            'frame': iframe_frame,
                            'form_count': len(form_elements)
                        }
                        break
                        
                except Exception as e:
                    dprint(f"⚠️ Error checking iframe {i+1}: {e}")
                    continue
            
            if iframe_with_forms:
                dprint(f"🎯 Using iframe context for form fields")
                return {
                    'has_iframes': True,
                    'iframe_count': len(visible_iframes),
                    'use_iframe_context': True,
                    'iframe_context': iframe_with_forms
                }
            else:
                dprint("ℹ️ No iframes with form elements found - using main page context")
                return {
                    'has_iframes': True,
                    'iframe_count': len(visible_iframes),
                    'use_iframe_context': False,
                    'iframe_context': None
                }
                
        except Exception as e:
            dprint(f"❌ Error detecting iframes: {e}")
            return {
                'has_iframes': False,
                'iframe_count': 0,
                'use_iframe_context': False,
                'iframe_context': None
            }
        
    def fill_all_form_inputs(self, text_fields: List[TextApplicationField], select_fields: List[SelectApplicationField], radio_fields: List[RadioApplicationField], checkbox_fields: List[CheckboxApplicationField], upload_fields: List[UploadApplicationField], frame=None, iframe_context=None) -> bool: 
        """Fill all form inputs using unified functions"""
        try:
            context_str = "iframe" if frame else "main page"
            print(f"🎯 Filling form fields in {context_str} context...")
            
            # Fill text input fields
            for field in text_fields:
                self.click_and_type_in_field(field, field.field_value, frame)
            
            # Fill select fields
            for field in select_fields:
                self.handle_select_field(field, frame)
            
            # Fill radio buttons
            for field in radio_fields:
                self.click_radio_button(field, frame)
            
            # Fill checkboxes
            for field in checkbox_fields:
                self.click_checkbox(field, frame)
            
            # Fill upload buttons
            for field in upload_fields:
                if field is not None:
                    self.handle_file_upload(field, frame, iframe_context)
            
            return True
        except Exception as e:
            print(f"❌ Error filling form inputs: {e}")
            traceback.print_exc()
            return False
    
    def handle_select_field(self, select_field: SelectApplicationField, frame=None) -> bool:
        """Handle select field in page or iframe context"""
        try:
            print(f"🔄 Handling select field: {select_field.field_name} - {select_field.field_selector}")
            # Use id= selector engine for IDs with special characters like []
            if select_field.field_selector.startswith('#'):
                if frame:
                    element = frame.locator(f'id={select_field.field_selector[1:]}').first
                else:
                    element = self.page.locator(f'id={select_field.field_selector[1:]}').first
            else:
                if frame:
                    element = frame.locator(select_field.field_selector)
                else:
                    element = self.page.locator(select_field.field_selector)
            
            if not element:
                print(f"⚠️ Select field not found with selector: {select_field.field_selector}")
                return False
            
            # Check if element is interactive before attempting to interact
            if not self._is_element_interactive(element, frame):
                context_str = "iframe" if frame else "page"
                print(f"⏭️ Skipping non-interactive select field in {context_str}: {select_field.field_name}")
                return True  # Return True to continue with other fields
            
            # Check if element is within form context before interacting
            if not self._is_element_in_form_context(element, frame):
                context_str = "iframe" if frame else "page"
                print(f"⏭️ Skipping select field outside form context in {context_str}: {select_field.field_name}")
                return True
            
            # Try to handle as standard select first
            try:
                # Try standard select handling
                possible_values = element.locator("option").evaluate_all(
                    "options => options.map(option => option.value)"
                )
                
                alternative_value = self._find_alternative_select_value(possible_values, select_field, self.preferences)
                select_value = alternative_value
                
                if possible_values:
                    if alternative_value:
                        print(f"🔄 Using alternative value: {alternative_value}")
                        element.select_option(value=str(select_value))
                        context_str = "iframe" if frame else "page"
                        print(f"✅ Successfully selected option in {context_str}: {select_field.field_name}")
                        return True
                    else:
                        print(f"❌ Could not find alternative value for {select_field.field_name}")
                        return False
                else:
                    # Alternative: Click the select field and then click the option that is the closest match to the value
                    print("🔄 Clicking select field and then clicking the option that is the closest match to the value")
                    element.click()
                    # self.page.pause()
                    time.sleep(0.5)
                    print(f"🔄 Clicked select field and waiting for options to appear")
                    
                    result = self._find_option_with_gpt(frame.content(), self.preferences, frame)

                    # Verify the result using the verify function
                    verified_result = self.verify_option_selector(result, frame.content(), self.preferences, best_of=3)
                    if verified_result and verified_result != result:
                        print(f"🔍 Selector verified and updated: {result} -> {verified_result}")
                        result = verified_result
                        
                    if result:
                        option_element = frame.locator(result)
                        option_element.click(force=True, timeout=5000)
                        print(f"✅ Successfully selected option: {result}")
                        return True
                    else:
                        print(f"❌ Could not find option element with selector: {result}")
                    return False
                
            except Exception as e:
                print(f"❌ Standard select handling failed: {e}")
                return False
                
        except Exception as e:
            context_str = "iframe" if frame else "page"
            print(f"❌ Error handling select field in {context_str}: {e}")
            return False

    def _find_option_with_gpt(self, page_html: str, preferences: str, frame) -> Optional[Dict[str, Any]]:
        """Use gpt-5-mini to find and click the correct option from HTML snapshot"""
        try:
            prompt = f"""
                You are **Gemini 2.5**, a state-of-the-art language model with DOM-parsing skills.

                **Task**  
                Given the raw HTML of an **open** *react-select* dropdown, output **only** the single, Playwright-Python selector that uniquely matches the user's preferences.  
                Return **nothing but that selector string** - no quotes, no back-ticks, no explanations.

                ------------------------------------------------------------------------
                Selection rules (follow in order; stop at the first one that yields a unique match)
                1. **Exact visible text**  
                `div[role="option"]:has-text("target_value")`
                2. **Autogenerated id** (stable within the page's lifetime)  
                Example: `#react-select-7-option-0`
                3. **Class ending in "-option"**  
                Example: `div.css-xyz789-option`
                4. **data-value** attribute  
                Example: `div[role="option"][data-value="target_value"]`
                5. If multiple candidates remain, prefer the shortest selector that is still unique.

                **Constraints**  
                • The selector must match **only the desired `<div role="option">` element** - never its parent listbox, descendants, or siblings.  
                • It must be a valid **Playwright Python** selector.  
                • Output nothing except the selector itself (no quotes, code fences, or commentary).  
                • Do **not** rely on brittle positional selectors (`nth-child`, `nth-of-type`, etc.) unless all other strategies fail.

                ------------------------------------------------------------------------
                ✅ **Allowed examples** (unique)  
                • `div[role="option"]:has-text("United Kingdom")`  
                • `#react-select-7-option-0`  
                • `div.css-a1b2c3-option[data-value="ca"]`  
                • `div[role="option"][data-value="fr"]`  

                ❌ **Forbidden examples** (non-unique or fragile)  
                • `div[role="option"]`                — matches every option  
                • `.css-123456`                       — class may apply to many nodes  
                • `div[role="option"]:first-child`   — position breaks if list changes  
                • `#react-select-3-option-`           — partial id, not unique
                • `#react-select-question_32214839002-option-0`           — partial id, not unique
                

                ------------------------------------------------------------------------
                **Return format**  
                Exactly the selector string, nothing else.

                HTML of the open dropdown:
                {page_html}
                """
            
            result = generate_text(
                prompt,
                system_prompt="You are an expert at analyzing HTML and finding dropdown options. Return only the selector that will find the option element.",
                model="gpt-5-mini"
            ).strip()
            
            print(result)
        
            return result
            
        except Exception as e:
            print(f"❌ Error in Gemini 2.5 Pro option detection: {e}")
            return None

    def verify_option_selector(self, selector_to_verify: str, page_html: str, preferences: str, best_of: int = 3) -> str:
        """
        Verify the option selector using best_of parameter for consensus
        """
        print(f"🔍 Verifying option selector: {selector_to_verify} (best_of: {best_of})")
        
        if best_of <= 1:
            # Single run - use original logic
            try:
                system_prompt = f"""
                    ROLE
                    You are a senior UI/DOM analyst. Verify a proposed Playwright CSS selector for a dropdown option element. If it is fully valid, return it UNCHANGED. If it fails any check, IGNORE it and RECOMPUTE the correct selector from the HTML. Output EXACTLY ONE line: the selector string or an empty string.

                    INPUTS
                    - proposed_selector: {selector_to_verify}
                    - page_html: HTML of the open dropdown
                    - preferences: {preferences}
                    - NOTE: {"The proposed selector is empty - you MUST find a dropdown option from scratch" if not selector_to_verify else "The proposed selector exists and should be validated"}

                    TARGET (WHAT COUNTS)
                    A single, visible dropdown option element that matches the user's preferences:
                    - Element is a <div role="option"> or similar dropdown option element
                    - Element text/content matches the user's preferences
                    - Element is visible and clickable
                    - Element is unique within the dropdown

                    FORBIDDEN / EXCLUDE
                    - Hidden/disabled/off-screen elements (display:none, visibility:hidden, opacity:0, [hidden], [aria-hidden="true"])
                    - Non-option elements (parent containers, siblings, descendants)
                    - Brittle selectors: :nth-*, inline style predicates, state-dependent hacks
                    - Overly generic selectors that match multiple elements

                    PREFERENCES FOR SELECTOR STABILITY
                    - Prefer stable attributes: #id, [data-*], [data-value], [aria-label], [title]
                    - Use :has-text() for text-based matching when stable attributes aren't available
                    - Prefer class-based selectors ending in "-option" for react-select components
                    - Avoid positional selectors unless absolutely necessary

                    VALIDATION CHECKLIST (APPLY TO proposed_selector)
                    1) Resolves to EXACTLY ONE element in the dropdown
                    2) Element has role="option" or similar dropdown option semantics
                    3) Element is visible & clickable now
                    4) Element text/content matches the user's preferences
                    5) Selector is reasonably stable and not overly brittle

                    DECISION RULE
                    - If ALL checklist items pass, RETURN proposed_selector UNCHANGED
                    - Otherwise, RECOMPUTE:
                    a) Locate the open dropdown container
                    b) Enumerate visible, clickable option elements within it
                    c) Filter by text/content matching the user's preferences
                    d) Build the most stable unique selector (id/data/aria + :has-text or class-based)
                    e) Ensure uniqueness, visibility, and proper role semantics

                    OUTPUT
                    - Output EXACTLY ONE line with ONLY the selector string (no quotes, no extra text)
                    - If no qualifying option found, output an empty string

                    REASONING PROTOCOL (DO NOT REVEAL)
                    - Validate proposed_selector with the checklist
                    - If invalid/ambiguous, recompute per procedure
                    - Final self-check: uniqueness, visibility, semantics, stability
                    """

                verified_selector = generate_text(
                    "You are an expert at analyzing HTML and finding dropdown options. Return only the selector that will find the option element.",
                    system_prompt=system_prompt,
                    model="gpt-5-mini"
                )
                
                print(f"🔍 Verified option selector: {verified_selector}")
                return verified_selector
            except Exception as e:
                print(f"❌ Error in verify_option_selector: {e}")
                return selector_to_verify
        
        # Multiple runs for consensus
        results = []
        for i in range(best_of):
            try:
                print(f"🔍 Run {i+1}/{best_of} for option selector verification...")
                
                system_prompt = f"""
                    ROLE
                    You are a senior UI/DOM analyst. Verify a proposed Playwright CSS selector for a dropdown option element. If it is fully valid, return it UNCHANGED. If it fails any check, IGNORE it and RECOMPUTE the correct selector from the HTML. Output EXACTLY ONE line: the selector string or an empty string.

                    INPUTS
                    - proposed_selector: {selector_to_verify}
                    - page_html: HTML of the open dropdown
                    - preferences: {preferences}
                    - NOTE: {"The proposed selector is empty - you MUST find a dropdown option from scratch" if not selector_to_verify else "The proposed selector exists and should be validated"}

                    TARGET (WHAT COUNTS)
                    A single, visible dropdown option element that matches the user's preferences:
                    - Element is a <div role="option"> or similar dropdown option element
                    - Element text/content matches the user's preferences
                    - Element is visible and clickable
                    - Element is unique within the dropdown

                    FORBIDDEN / EXCLUDE
                    - Hidden/disabled/off-screen elements (display:none, visibility:hidden, opacity:0, [hidden], [aria-hidden="true"])
                    - Non-option elements (parent containers, siblings, descendants)
                    - Brittle selectors: :nth-*, inline style predicates, state-dependent hacks
                    - Overly generic selectors that match multiple elements

                    PREFERENCES FOR SELECTOR STABILITY
                    - Prefer stable attributes: #id, [data-*], [data-value], [aria-label], [title]
                    - Use :has-text() for text-based matching when stable attributes aren't available
                    - Prefer class-based selectors ending in "-option" for react-select components
                    - Avoid positional selectors unless absolutely necessary

                    VALIDATION CHECKLIST (APPLY TO proposed_selector)
                    1) Resolves to EXACTLY ONE element in the dropdown
                    2) Element has role="option" or similar dropdown option semantics
                    3) Element is visible & clickable now
                    4) Element text/content matches the user's preferences
                    5) Selector is reasonably stable and not overly brittle

                    DECISION RULE
                    - If ALL checklist items pass, RETURN proposed_selector UNCHANGED
                    - Otherwise, RECOMPUTE:
                    a) Locate the open dropdown container
                    b) Enumerate visible, clickable option elements within it
                    c) Filter by text/content matching the user's preferences
                    d) Build the most stable unique selector (id/data/aria + :has-text or class-based)
                    e) Ensure uniqueness, visibility, and proper role semantics

                    OUTPUT
                    - Output EXACTLY ONE line with ONLY the selector string (no quotes, no extra text)
                    - If no qualifying option found, output an empty string

                    REASONING PROTOCOL (DO NOT REVEAL)
                    - Validate proposed_selector with the checklist
                    - If invalid/ambiguous, recompute per procedure
                    - Final self-check: uniqueness, visibility, semantics, stability
                    """

                result = generate_text(
                    "You are an expert at analyzing HTML and finding dropdown options. Return only the selector that will find the option element.",
                    system_prompt=system_prompt,
                    model="gpt-5-mini"
                )
                
                if result and result.strip():
                    results.append(result.strip())
                    print(f"🔍 Run {i+1} result: {result.strip()}")
                else:
                    print(f"🔍 Run {i+1} returned empty result")
                    
            except Exception as e:
                print(f"❌ Error in run {i+1}: {e}")
                continue
        
        if not results:
            print(f"❌ All {best_of} runs failed, returning original selector")
            return selector_to_verify
        
        # Find most common result
        from collections import Counter
        result_counts = Counter(results)
        most_common_result = result_counts.most_common(1)[0]
        
        print(f"🔍 Consensus results: {dict(result_counts)}")
        print(f"🔍 Most common result: {most_common_result[0]} (frequency: {most_common_result[1]}/{best_of})")
        
        # If most common result appears more than once and is different from original, use it
        if most_common_result[1] > 1 and most_common_result[0] != selector_to_verify:
            print(f"🔍 Using consensus result: {most_common_result[0]}")
            return most_common_result[0]
        else:
            print(f"🔍 No clear consensus, returning last result: {results[-1] if results else selector_to_verify}")
            return results[-1] if results else selector_to_verify
    
    def _check_select_already_selected(self, element, target_value: str, frame=None) -> bool:
        """
        Check if a select field already has the target value selected
        """
        try:
            if not target_value:
                return False
            
            target_lower = target_value.lower().strip()
            
            # For traditional select elements
            try:
                selected_option = element.evaluate("el => el.selectedOptions[0]?.value || el.selectedOptions[0]?.text")
                if selected_option:
                    selected_lower = selected_option.lower().strip()
                    if selected_lower == target_lower:
                        return True
            except:
                pass
            
            # For custom dropdowns, check inner text (but be more strict)
            try:
                inner_text = element.inner_text().strip()
                if inner_text:
                    inner_lower = inner_text.lower().strip()
                    # Only consider it selected if it's an exact match or very close match
                    if inner_lower == target_lower or (len(inner_lower) > 0 and inner_lower in target_lower and len(inner_lower) > 2):
                        # Additional check - make sure it's not a placeholder
                        if not any(placeholder in inner_lower for placeholder in ["select", "choose", "pick", "option", "please", "..."]):
                            return True
            except:
                pass
            
            # Check for data attributes that might indicate selection
            try:
                data_selected = element.get_attribute("data-selected")
                if data_selected and data_selected.lower().strip() == target_lower:
                    return True
                
                value_attr = element.get_attribute("value")
                if value_attr and value_attr.lower().strip() == target_lower:
                    return True
            except:
                pass
            
            return False
            
        except Exception as e:
            print(f"⚠️ Error checking if select already selected: {e}")
            return False

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
                form_ancestor = context.locator('xpath=ancestor::form').first
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

    def click_radio_button(self, radio_button: RadioApplicationField, frame=None) -> bool:
        """Click on a radio button in page or iframe context"""
        try:
            # Use id= selector engine for IDs with special characters like []
            if radio_button.field_selector.startswith('#'):
                if frame:
                    element = frame.locator(f'id={radio_button.field_selector[1:]}').first
                else:
                    element = self.page.locator(f'id={radio_button.field_selector[1:]}').first
            else:
                if frame:
                    element = frame.locator(f"input[type='radio'][name='{radio_button.field_name}'][value='{radio_button.field_value}']")
                else:
                    element = self.page.locator(f"input[type='radio'][name='{radio_button.field_name}'][value='{radio_button.field_value}']")
            
            if not element:
                print(f"⚠️ Radio button not found with selector: {radio_button.field_selector}")
                return False
            
            # Check if element is interactive before attempting to click
            if not self._is_element_interactive(element, frame):
                context_str = "iframe" if frame else "page"
                print(f"⏭️ Skipping non-interactive radio button in {context_str}: {radio_button.field_name}")
                return True  # Return True to continue with other fields
                
            radio_value: bool = radio_button.field_value
            if radio_value:
                element.check()
                context_str = "iframe" if frame else "page"
                print(f"✅ Clicked radio button in {context_str}")
            return True
        except Exception as e:
            context_str = "iframe" if frame else "page"
            print(f"❌ Error checking radio button in {context_str}: {e}")
            return False

    def click_checkbox(self, checkbox: CheckboxApplicationField, frame=None) -> bool:
        """Click on a checkbox in page or iframe context"""
        try:
            # Use id= selector engine for IDs with special characters like []
            if checkbox.field_selector.startswith('#'):
                if frame:
                    element = frame.locator(f'id={checkbox.field_selector[1:]}').first
                else:
                    element = self.page.locator(f'id={checkbox.field_selector[1:]}').first
            else:
                if frame:
                    element = frame.locator(checkbox.field_selector)
                else:
                    element = self.page.locator(checkbox.field_selector)
            
            if not element:
                print(f"⚠️ Checkbox not found with selector: {checkbox.field_selector}")
                return False
            
            # Check if element is interactive before attempting to click
            if not self._is_element_interactive(element, frame):
                context_str = "iframe" if frame else "page"
                print(f"⏭️ Skipping non-interactive checkbox in {context_str}: {checkbox.field_selector}")
                return True  # Return True to continue with other fields
                
            checkbox_value: bool = checkbox.field_value
            if checkbox_value:
                element.click()
                context_str = "iframe" if frame else "page"
                print(f"✅ Clicked checkbox in {context_str}")
            return True
        except Exception as e:
            context_str = "iframe" if frame else "page"
            print(f"❌ Error clicking checkbox in {context_str}: {e}")
            return False

    def handle_file_upload(self, upload_button: UploadApplicationField, frame=None, iframe_context=None, times_tried=0) -> bool:
        """Handle file upload in page or iframe context"""
        try:
            # Use id= selector engine for IDs with special characters like []
            if upload_button.field_selector.startswith('#'):
                if frame:
                    element = frame.locator(f'id={upload_button.field_selector[1:]}').first
                else:
                    element = self.page.locator(f'id={upload_button.field_selector[1:]}').first
            else:
                if frame:
                    element = frame.locator(upload_button.field_selector)
                else:
                    element = self.page.locator(upload_button.field_selector)
            
            if not element:
                print(f"⚠️ Upload button not found with selector: {upload_button.field_selector}")
                return False
            
            # Check if element is within form context before interacting
            if not self._is_element_in_form_context(element, frame):
                context_str = "iframe" if frame else "page"
                print(f"⏭️ Skipping upload button outside form context in {context_str}: {upload_button.field_name}")
                return True
            
            try:
                element.click()
                self.page.pause()
                
                context_str = "iframe" if frame else "page"
                print(f"✅ Successfully triggered file dialog in {context_str}")
                return True
            except Exception as e:
                print(f"⚠️ File dialog triggering failed: {e}")
            
            # If all methods failed and we haven't tried too many times, retry
            if times_tried < 3:
                print(f"🔄 Retrying file upload (attempt {times_tried + 1}/3)")
                time.sleep(1)
                
                # Find new upload button
                upload_button = self.find_upload_file_button(frame, iframe_context)
                return self.handle_file_upload(upload_button, frame, times_tried + 1)
            
            context_str = "iframe" if frame else "page"
            print(f"❌ Failed to upload file in {context_str} after {times_tried + 1} attempts")
            
            return False
            
        except Exception as e:
            context_str = "iframe" if frame else "page"
            print(f"❌ Error handling file upload in {context_str}: {e}")
            return False

    def find_submit_button_with_gpt(self, frame: Dict[str, Any] = None, iframe_context: Dict[str, Any] = None) -> SubmitButtonApplicationField:
        """
        Use gpt-5-mini to contextually find submit buttons on the page or in iframe
        
        Args:
            iframe_context: Iframe context if searching within an iframe
            
        Returns:
            SubmitButtonApplicationField with submit button information and action
        """
        try:
            print("🤖 Using gpt-5-mini to find submit button contextually...")
            
            if frame:
                page_content = frame.content()
            else:
                page_content = self.page.content()
            
            # Use smart screenshot that handles iframes
            screenshot, context_info = self._take_smart_screenshot(frame, iframe_context)
            if screenshot is None:
                print(f"❌ Could not take screenshot, using fallback analysis")
                return SubmitButtonApplicationField(field_type=FieldType.SUBMIT, field_selector=None, field_is_visible=False, field_in_form=False, field_required=False)

            prompt = f"""
            
                1.	TASK
            Locate the single HTML element that submits or advances a job-application form and return it as an instantiation of SubmitButtonApplicationField.
             
            You are also given a screenshot of the page. Use the screenshot to help you identify the fields.
            If there is a modal or dialog, only focus on the fields that are visible in the modal and not grayed out.
            
                2.	OUTPUT CONTRACT — return ONLY this Python snippet, nothing else
            SubmitButtonApplicationField(
            field_type       = FieldType.SUBMIT,
            field_selector   = ,
            field_is_visible = <True | False | None>,
            field_in_form    = <True | False | None>,
            field_required   = <True | False | None>
            )
            If no suitable element exists, set field_selector = None and all booleans to False.
                3.	SELECTION RULES
            • Allowed element types: , , or any tag with role="button".
            • Text cues (case-insensitive): submit, continue, next, apply, send, save, finish, complete.
            • Attribute cues: type="submit"; class/id containing a text cue; data-action*="submit"; onclick that submits a form or advances the process.
            • Preference hierarchy:

                1.	Prefer elements that are descendants of the primary .
                2.	If multiple candidates exist within that form, choose the one nearest the end of the form's content flow.
                3.	If ties remain, prefer elements with explicit type="submit" or a text cue from the list.

                4.	FIELD VALUE GUIDANCE
            • field_selector   — stable, concise CSS selector (id if present, otherwise a specific path).
            • field_is_visible — True if element is not hidden (no hidden attribute, not aria-hidden="true", not style/display:none, not visibility:hidden); False if hidden; None if unknown.
            • field_in_form    — True if the element is inside a ; False otherwise; None if unknown.
            • field_required   — True only if the element explicitly has required or aria-required="true"; otherwise False or None if unknown.
                5.	IN-CONTEXT EXAMPLES

            GOOD G1 (inside form, explicit submit)
            HTML:
            <form id="app">
            …
            <button type="submit" class="btn primary">Apply Now</button>
            </form>
            Python:
            SubmitButtonApplicationField(
            field_type       = FieldType.SUBMIT,
            field_selector   = "#app button[type='submit']",
            field_is_visible = True,
            field_in_form    = True,
            field_required   = False
            )

            GOOD G2 (role=button with submit cue)
            HTML:
            <form>
            …
            <a role="button" id="continue" class="next">Continue</a>
            </form>
            Python:
            SubmitButtonApplicationField(
            field_type       = FieldType.SUBMIT,
            field_selector   = "a#continue.next",
            field_is_visible = True,
            field_in_form    = True,
            field_required   = False
            )

            BAD B1 (not a submit element)
            HTML:
            <a href="/privacy">Read our privacy policy</a>
            Python:
            SubmitButtonApplicationField(
            field_type       = FieldType.SUBMIT,
            field_selector   = None,
            field_is_visible = False,
            field_in_form    = False,
            field_required   = False
            )

            BAD B2 (multiple candidates — do NOT pick arbitrarily)
            HTML:
            <form>
            <button>Back</button>
            <button>Next</button>
            <button>Save Draft</button>
            </form>
            Correct behaviour: choose the element that advances the flow ("Next"); if ambiguous, pick the one nearest the end of the form.
                6.	INPUT
            HTML source:
            {page_content}
            """
            
            result = generate_model(
                prompt,
                model_object_type=SubmitButtonApplicationField,
                system_prompt="You are an expert at analyzing HTML and identifying submit buttons for job application forms. Return only valid JSON.",
                image=screenshot,
                model="gpt-5-mini"
            )
            
            print(f"🤖 gpt-5-mini found submit button: {result}")
            
            return result
            
        except Exception as e:
            print(f"❌ Error in Gemini submit button detection: {e}")
            return None

    def _find_and_click_submit_button_fallback(self) -> bool:
        """
        Fallback method to find and click submit button using traditional selectors
        
        Returns:
            bool: True if button was found and clicked successfully
        """
        try:
            print("🔍 Using traditional selectors to find submit button...")
            
            # Common submit button selectors
            submit_selectors = [
                'button[type="submit"]',
                'input[type="submit"]',
                'button:has-text("Submit")',
                'button:has-text("Continue")',
                'button:has-text("Next")',
                'button:has-text("Apply")',
                'button:has-text("Send")',
                '[data-testid*="submit"]',
                '[data-testid*="continue"]',
                '[data-testid*="next"]',
                '[class*="submit"]',
                '[class*="continue"]',
                '[id*="submit"]',
                '[id*="continue"]'
            ]
            
            # First, try to find submit button in main page
            submit_button = None
            for selector in submit_selectors:
                try:
                    buttons = self.page.query_selector_all(selector)
                    for button in buttons:
                        if button.is_visible() and button.is_enabled():
                            submit_button = button
                            print(f"✅ Found submit button in main page with selector: {selector}")
                            break
                    if submit_button:
                        break
                except:
                    continue
            
            # If not found in main page, check iframes
            if not submit_button:
                print("🔍 Checking iframes for submit button...")
                iframes = self.page.query_selector_all('iframe')
                visible_iframes = [iframe for iframe in iframes if iframe.is_visible()]
                
                for i, iframe in enumerate(visible_iframes):
                    try:
                        iframe_frame = iframe.content_frame()
                        if not iframe_frame:
                            continue
                        
                        # Check iframe for submit buttons
                        for selector in submit_selectors:
                            try:
                                buttons = iframe_frame.query_selector_all(selector)
                                for button in buttons:
                                    if button.is_visible() and button.is_enabled():
                                        submit_button = button
                                        print(f"✅ Found submit button in iframe {i+1} with selector: {selector}")
                                        break
                                if submit_button:
                                    break
                            except:
                                continue
                        
                        if submit_button:
                            break
                            
                    except Exception as e:
                        print(f"⚠️ Error checking iframe {i+1} for submit button: {e}")
                        continue
            
            if not submit_button:
                print("❌ No submit button found in main page or iframes")
                return False
            
            # Click the submit button
            print("🎯 Clicking submit button...")
            if submit_button.is_visible():
                submit_button.click()
                time.sleep(3)
                print("✅ Submit button clicked")
                return True
            else:
                print("🎯 No submit button found, finding next button...")
                return self.find_and_click_next_button()
            
        except Exception as e:
            print(f"❌ Error finding/clicking submit button: {e}")
            return False
    
    def find_and_click_next_button(self) -> bool:
        """
        Find and click the next button
        """
        # Try multiple selectors for next/continue buttons
        next_selectors = [
            "button:has-text('Next')",
            "button:has-text('Continue')",
            "button:has-text('next')",
            "button:has-text('continue')",
            "[data-testid*='next']",
            "[data-testid*='continue']",
            "[class*='next']",
            "[class*='continue']",
            "input[value*='Next']",
            "input[value*='Continue']"
        ]
        
        for selector in next_selectors:
            try:
                next_button = self.page.locator(selector)
                if next_button.is_visible():
                    next_button.click()
                    time.sleep(3)
                    print(f"✅ Clicked next button with selector: {selector}")
                    return True
            except Exception as e:
                print(f"⚠️ Error with selector {selector}: {e}")
                continue
        
        print("❌ No next/continue button found")
        return False
  
    def check_form_submission_with_gpt(self, initial_screenshot: bytes, frame: Dict[str, Any] = None, iframe_context: Dict[str, Any] = None) -> ApplicationStateResponse:
        """
        Use gpt-5-mini to analyze if the form was successfully submitted
        
        Args:
            frame: Frame to analyze
            
        Returns:
            ApplicationStateResponse with submission status and categorized fields
        """
        try:
            # For now, use page content analysis instead of image analysis
            # In a real implementation, you could use Gemini 2.5 Pro API
            
            if frame:
                page_content = frame.content()
            else:
                page_content = self.page.content()
                
            time.sleep(5)
            screenshot, _ = self._take_smart_screenshot(frame, iframe_context)
            
            prompt = f"""
                Analyse the web page below and output the best result.

                You are given the screenshot of the page. Use this in contrast to the change in the HTML to determine the best result.
                You are also given the HTML of the page. Use the HTML to determine the best result.
                
                {{
                "submitted":            bool,  # form data accepted, look for "thank you" "application submitted" "confirmation URLs"
                "completed":            bool,  # application fully finished
                "error_in_submission":  bool,  # any submission/validation/server error present
                "verification_required":bool,  # login/OTP/captcha
                "more_forms":           bool,   # additional steps or forms still visible
                "reason":               str     # reason for the result
                }}
                
                One of the fields must be true.
                
                The HTML of the page: {page_content}
                """
            
            result = generate_model(
                prompt,
                model_object_type=ApplicationStateResponse,
                system_prompt="You are an expert at analyzing job application pages. Return only valid JSON.",
                image=screenshot,
                model="gpt-5-mini"
            )
            
            # Parse JSON response
            try:
                print(f"🤖 Gemini 2.5 Pro analysis - Submitted: {result.submitted}, Completed: {result.completed}, Verification required: {result.verification_required}, More forms: {result.more_forms}, Error in submission: {result.error_in_submission}")
                print(f"🤖 Gemini 2.5 Pro analysis - Reason: {result.reason}")
                return result
                
            except json.JSONDecodeError:
                print(f"⚠️ Could not parse Gemini response: {result}")
                return ApplicationStateResponse(submitted=False, completed=False, verification_required=False, more_forms=False, error_in_submission=False)
                
        except Exception as e:
            print(f"❌ Error in Gemini submission analysis: {e}")
            return ApplicationStateResponse(submitted=False, completed=False, verification_required=False, more_forms=False, error_in_submission=False)

    def find_all_text_input_fields(self, frame=None, iframe_context=None, spinner:Yaspin=None) -> List[TextApplicationField]:
        """
        Find all text input fields in page or iframe context
        
        Args:
            frame: Optional iframe frame context. If None, uses main page.
        """
        try:
            # Get page content from frame or main page
            if frame:
                page_content = frame.content()
            else:
                page_content = self.page.content()

            # Use gpt-5-mini to analyze the page and identify form fields
            system_prompt = f"""
            ROLE
            You are a senior UI/DOM form analyst. Extract ONLY genuine text-entry fields from a job application experience and return strict JSON for Python Playwright. Also populate each field's value using the provided preferences.

            INPUTS
            - preferences: {self.preferences}
            - html: {page_content}
            - screenshot: current UI state

            REASONING PROTOCOL (DO NOT REVEAL)
            1) Determine the active scope: if a visible modal/dialog is open (role='dialog' or [aria-modal='true']), restrict to that modal; otherwise use the page form(s).
            2) Enumerate candidate controls in DOM order.
            3) Filter to text-entry controls only; exclude selects, radios, checkboxes, file inputs, pseudo-dropdowns, and hidden/disabled/off-screen/future-step content.
            4) For each remaining control, derive a minimal, stable selector and map a value from preferences.
            5) Self-check: uniqueness, visibility, scope, and correct control type.
            6) Output ONLY the JSON array.

            OBJECTIVE
            Identify every VISIBLE, INTERACTIVE text-entry field used to type information and fill it with values from preferences.

            ELIGIBLE CONTROLS
            - <input> types: text, email, tel, url, search, number, password, date, datetime-local, time
            - <textarea>
            Notes:
            - <input list="..."> (with <datalist>) is allowed (still text entry).
            - Contenteditable widgets are EXCLUDED unless they have a stable input/textarea-equivalent role and are clearly used for text entry (generally avoid).

            INELIGIBLE / EXCLUDE
            - <select>, React-Select controls, menus, options
            - <input type='checkbox' | 'radio' | 'file' | 'range' | 'color' | 'hidden'>
            - Pseudo-dropdowns/autocomplete pickers that are primarily selection UIs:
            - input[role='combobox'] that controls a listbox popup and behaves as a selector
            - inputs inside React-Select (e.g., .select__control input)
            - Elements that are disabled, hidden, off-screen, zero-size, aria-hidden='true', or in non-active steps/tabs/accordions
            - Anything outside the job application form scope
            - Cross-document selectors (iframes in the selector path). If fields are inside an iframe, return a selector that works *within that iframe context* (no iframe prefix).

            SCOPE & VISIBILITY RULES
            - Only include fields inside the job application form (a <form> ancestor clearly tied to applying).
            - If a visible modal/dialog is open, treat it as the only active scope; ignore background page fields.
            - Exclude future-step or collapsed sections unless explicitly expanded/active (e.g., class contains 'open' or 'active').

            SELECTOR RULES (Playwright-compatible CSS)
            - Target the ACTUAL control element (<input> or <textarea>) directly.
            - Prefer stable attributes and scoping: name, id, data-*, aria-*, autocomplete, and clear form/field containers.
            - Avoid brittle patterns: :nth-*, positional indices, transient state selectors ([aria-expanded], inline styles), overly generic roots (html body ...).
            - Do NOT include iframe nodes in the selector string.

            GOOD SELECTORS (examples)
            input[name='first_name']
            input#email
            input[type='email']
            textarea[name='cover_letter']
            form#application-form input[name='phone']
            form[aria-label='Job application'] input[data-qa='city']
            form#apply-modal-form input#address_line1
            form.ApplicationForm input[autocomplete='postal-code']
            form#application-form section.step-current input[name='expected_salary']

            BAD SELECTORS (and why)
            input[type='checkbox']                 # not text entry
            input[type='radio']                    # not text entry
            select[name='country']                 # dropdown, not text entry
            div.select__control input              # part of React-Select, selection UI
            input[disabled]                        # not interactive
            [style*='display:none'] input          # hidden
            iframe#grnhse_iframe input[name='x']   # crosses iframe in selector
            html body input                        # overly generic
            input:nth-of-type(2)                   # positional, brittle

            PREFERENCES → VALUE MAPPING (best-effort; DO NOT HALLUCINATE)
            - Infer field intent from name/id/label/placeholder/aria-label and nearby text in the screenshot.
            - Common mappings (case-insensitive substring or regex on field identifiers):
            - first_name, given_name → preferences['first_name']
            - last_name, family_name, surname → preferences['last_name']
            - full_name, name → preferences['full_name'] or first_name + ' ' + last_name (if both present)
            - email, email_address → preferences['email']
            - phone, phone_number, mobile → preferences['phone']
            - address, address1, street → preferences['address_line1']
            - address2, apartment, unit → preferences.get('address_line2', '')
            - city, town → preferences['city']
            - state, province, region → preferences['state']
            - postal, postcode, zip → preferences['postal_code']
            - country → preferences['country']
            - linkedin → preferences['linkedin']
            - github → preferences['github']
            - website, url, portfolio → preferences['website']
            - cover_letter, motivation, summary → preferences['cover_letter']
            - salary, expected_salary → preferences['expected_salary']
            - dob, date_of_birth, birthday → preferences['date_of_birth']
            - If no suitable preference exists, set field_value to "" (empty string).
            - Do not invent formats; apply minimal normalization only when obvious (e.g., trim spaces).

            OUTPUT SCHEMA (STRICT JSON ARRAY)
            Return ONLY a JSON array of objects. Each object MUST have:
            - "field_selector": minimal, stable CSS selector for the <input>/<textarea>
            - "field_name": the name attribute or best-effort visible label (from HTML/screenshot)
            - "field_type": one of "text", "email", "tel", "url", "search", "number", "password", "date", "datetime-local", "time", "textarea"
            - "field_is_visible": true only if visible and interactive now
            - "field_in_form": true only if inside the application form (or active modal form)
            - "field_value": the value from preferences or "" if unavailable

            ORDERING
            - Preserve top-to-bottom DOM order within the active scope.

            VALIDATION & SELF-CHECK (DO NOT OUTPUT)
            - Each selector resolves to exactly one eligible control of the correct type.
            - No item is hidden/disabled/off-screen or outside the active application scope.
            - No dropdowns/checkboxes/radios/files/react-select inputs.
            - No iframe traversal in selectors.
            - If no qualifying fields, output [].

            FINAL OUTPUT REQUIREMENT
            - Output ONLY the JSON array per the schema above. No extra text.
            """

            
            all_fields = generate_model(
                "You are an expert at analyzing HTML and identifying form inputs. Return only valid JSON with CSS selectors.",
                model_object_type=TextApplicationFieldResponse,
                system_prompt=system_prompt,
                model="gpt-5-mini"
            ).fields
            
            all_fields = [field for field in all_fields if field.field_is_visible and field.field_in_form]
            
            # Parse the JSON response
            text_input_fields: List[TextApplicationField] = [field for field in all_fields if field.field_type == FieldType.TEXT]
            
            # verified_text_input_fields = self.verify_text_input_fields(text_input_fields, frame, iframe_context, spinner)
            
            context_str = "iframe" if frame else "page"
            dprint(f"✅ gpt-5-mini found {len(text_input_fields)} text input fields in {context_str}")
            
            return text_input_fields
            
        except Exception as e:
            print(f"❌ Error in find_all_text_input_fields: {e}")
            return []

    def find_all_radio_fields(self, frame=None, iframe_context=None, spinner:Yaspin=None) -> List[RadioApplicationField]:
        """
        Find all radio fields in page or iframe context
        
        Args:
            frame: Optional iframe frame context. If None, uses main page.
        """
        try:
            # Get page content from frame or main page
            if frame:
                page_content = frame.content()
            else:
                page_content = self.page.content()
                
            # Use gpt-5-mini to analyze the page and identify form fields
            system_prompt = f"""
            ROLE
            You are a senior UI/DOM form analyst. Identify radio-button fields for a job application and return ONLY the radio options that should be selected (true) as strict JSON for Python Playwright. Populate selections using the provided preferences and current UI state.

            INPUTS
            - preferences: {self.preferences}
            - html: {page_content}

            REASONING PROTOCOL (DO NOT REVEAL)
            1) Determine the active scope: if a visible modal/dialog is open (role='dialog' or [aria-modal='true']), restrict to that modal; otherwise use the page form(s).
            2) Enumerate radio groups by shared name attribute (input[type='radio'][name]).
            3) For each group, identify the single option that should be selected based on preferences and the visible label/value.
            4) Keep only visible, enabled, in-scope options; exclude hidden/disabled/off-screen/future-step content.
            5) Produce a minimal, stable selector for the chosen radio input.
            6) Self-check: at most one selection per group; correct control type; uniqueness; scope.
            7) Output ONLY the JSON array.

            OBJECTIVE
            Return the subset of radio inputs that should be selected now (true). Do not return unchecked options or entire groups—only the specific option to set true.

            ELIGIBLE CONTROLS
            - <input type="radio"> elements that belong to the job-application form and are currently visible & interactive.

            INELIGIBLE / EXCLUDE
            - checkboxes, selects, text inputs, file/range/color/hidden, React-Select internals
            - any hidden, disabled, off-screen, zero-size, aria-hidden='true', or in non-active steps/tabs/accordions
            - anything outside the job-application form scope
            - cross-document selectors (iframes in the selector path). If radios are inside an iframe, return a selector that works *within that iframe context* (no iframe prefix).

            SCOPE & VISIBILITY RULES
            - Only include fields inside the job application form (a <form> ancestor clearly tied to applying).
            - If a visible modal/dialog is open, treat it as the only active scope; ignore background page fields.
            - Exclude future-step or collapsed sections unless explicitly expanded/active (e.g., class contains 'open' or 'active').

            SELECTOR RULES (Playwright-compatible CSS)
            - Target the ACTUAL radio input element directly: input[type='radio'][name=...][value=...] whenever possible.
            - Prefer stable attributes and scoping: name, id, value, data-*, aria-*, and clear form/field containers.
            - Avoid brittle patterns: :nth-*, positional indices, transient-state selectors ([aria-expanded], inline styles), overly generic roots (html body ...).
            - Do NOT include iframe nodes in the selector string.

            GOOD SELECTORS (examples)
            input[type='radio'][name='work_authorization'][value='yes']
            form#application-form input[type='radio'][name='willing_to_relocate'][value='true']
            form[aria-label='Job application'] input[type='radio'][name='sponsorship_required'][value='no']
            #apply-modal.open form#apply-modal-form input[type='radio'][name='eu_citizen'][value='1']

            BAD SELECTORS (and why)
            label:has-text('Yes')                          # targets label, not the input
            input[type='checkbox'][name='relocate']        # not a radio
            [style*='display:none'] input[type='radio']    # hidden
            iframe#grnhse_iframe input[type='radio'][name='x']  # crosses iframe in selector
            html body input[type='radio']                  # overly generic
            input[type='radio']:nth-of-type(2)             # positional, brittle

            PREFERENCES → OPTION SELECTION (DO NOT HALLUCINATE)
            - Determine each group's intent from name/id/fieldset legend/label/aria-label and nearby text in the screenshot.
            - Build a canonical label for each option (visible label text, aria-label, or value attribute).
            - Map preferences to groups using case-insensitive substring/regex on likely keys. Examples:
            - work_authorization / work_auth / can_work / right_to_work → yes/no
            - sponsorship_required / need_sponsorship / visa_sponsorship → yes/no
            - willing_to_relocate / relocate / relocation → yes/no
            - remote_ok / open_to_remote / onsite_only → yes/no
            - eu_citizen / uk_citizen / us_citizen → yes/no
            - Normalize "truthy/yes" synonyms: yes, y, true, 1, accept, authorized, eligible, allow
            - Normalize "false/no" synonyms: no, n, false, 0, deny, not authorized, ineligible
            - For non-binary groups (multiple options), pick the option whose canonical text best matches the preference string (exact/substring/regex). If ambiguous or no suitable preference exists, SKIP the group (do not guess).
            - Never select more than one option per name group.

            OUTPUT SCHEMA (STRICT JSON ARRAY)
            Return ONLY a JSON array of objects. Each object MUST have:
            - "field_selector": minimal, stable CSS selector for the chosen <input type='radio'> option
            - "field_name": the radio group name (the input[name])
            - "field_type": "radio"
            - "field_is_visible": true only if visible and interactive now
            - "field_in_form": true only if inside the application form (or active modal form)
            - "field_value": true

            ORDERING
            - Preserve top-to-bottom DOM order of the chosen options within the active scope.

            VALIDATION & SELF-CHECK (DO NOT OUTPUT)
            - Each selector resolves to exactly one <input type='radio'>.
            - Exactly one selected option per radio group name; others omitted.
            - No item is hidden/disabled/off-screen or outside the active application scope.
            - No iframe traversal in selectors.
            - If no qualifying selections, output [].

            FINAL OUTPUT REQUIREMENT
            - Output ONLY the JSON array per the schema above. No extra text.
            """

            
            all_fields = generate_model(
                "You are an expert at analyzing HTML and identifying form inputs. Return only valid JSON with CSS selectors.",
                model_object_type=RadioApplicationFieldResponse,
                system_prompt=system_prompt,
                model="gpt-5-mini"
            ).fields
            
            all_fields = [field for field in all_fields if field.field_is_visible and field.field_in_form]
            
            context_str = "iframe" if frame else "page"
            
            # Parse the JSON response
            radio_fields: List[RadioApplicationField] = [field for field in all_fields if field.field_type == FieldType.RADIO]
            verified_radio_fields = self.verify_radio_fields(radio_fields, frame, iframe_context, spinner)
            dprint(f"✅ gpt-5-mini found {len(radio_fields)} radio fields in {context_str}")
            
            return verified_radio_fields
            
        except Exception as e:
            print(f"❌ Error in find_all_radio_fields: {e}")
            return []

    def find_all_checkbox_fields(self, frame=None, iframe_context=None, spinner:Yaspin=None) -> List[CheckboxApplicationField]:
        """
        Find all checkbox fields in page or iframe context
        
        Args:
            frame: Optional iframe frame context. If None, uses main page.
        """
        try:
            # Get page content from frame or main page
            if frame:
                page_content = frame.content()
            else:
                page_content = self.page.content()
            
            # Use gpt-5-mini to analyze the page and identify form fields
            system_prompt = f"""
                ROLE
                You are a senior UI/DOM form analyst. Identify ONLY checkbox controls for a job application and return strict JSON for Python Playwright. 

                INPUTS
                - preferences (dict-like): {self.preferences}
                - html: {page_content}
                
                For each checkbox, set field_value **using preferences only**.

                REASONING PROTOCOL (DO NOT REVEAL)
                1) Determine active scope: if a visible modal/dialog is open (role='dialog' or [aria-modal='true']), restrict to that; otherwise use the page form(s).
                2) Enumerate candidate inputs in DOM order.
                3) Keep ONLY input[type='checkbox'] that are visible, enabled, and inside the job-application form scope.
                4) Derive a minimal, stable selector for each checkbox.
                5) Compute field_value **from preferences only**:
                - Extract candidate text for the checkbox: name, id, value, label text, aria-label, and fieldset legend (all lowercased, snake_cased, punctuation stripped).
                - Normalize preferences: lowercased, snake_cased keys; normalize boolean-like values:
                    truthy → yes,y,true,1,on,accept,agree,consent,opt_in,subscribe,enabled
                    falsy  → no,n,false,0,off,decline,disagree,opt_out,unsubscribe,disabled
                - Matching priority (first hit wins):
                    a) Exact key match to checkbox name/id/value
                    b) Exact key match to normalized label/aria-label/legend
                    c) Substring/regex match of preference keys within the concatenated candidate text
                - If multiple preference keys match with conflicting booleans:
                    • Prefer exact match over substring; if still tied, prefer name/id/value over label/aria; if still tied, set false.
                - If **no** preference key matches, set field_value = false. Do **not** infer from label semantics or requirements.
                6) Self-check: selector resolves to exactly one input[type='checkbox']; is visible; in active scope; not crossing iframes.
                7) Output ONLY the JSON array.

                OBJECTIVE
                Return all VISIBLE, INTERACTIVE checkboxes relevant to the application, each with an explicit boolean field_value derived **solely from preferences**.

                ELIGIBLE CONTROLS
                - <input type="checkbox"> elements within the job-application form and currently visible & interactive.
                - Checkbox groups (same name, different value): include one object per actual checkbox input; match each to preferences by value/label text; otherwise false.

                INELIGIBLE / EXCLUDE
                - Radios, selects, text inputs, file/range/color/hidden, React-Select internals, custom toggles without a backing checkbox input
                - Hidden/disabled/off-screen/zero-size/aria-hidden='true'/future-step or inactive tab content
                - Anything outside the job-application form scope
                - Cross-document selectors (no iframe prefixes). If inside an iframe, return selectors that work within that iframe context only.

                SCOPE & VISIBILITY RULES
                - Only include fields inside a <form> clearly tied to applying.
                - If a visible modal/dialog is open, treat it as the only active scope; ignore background page fields.
                - Exclude collapsed/latent steps unless visibly active (e.g., class contains 'open' or 'active').

                SELECTOR RULES (Playwright-compatible CSS)
                - Target the ACTUAL checkbox input directly: input[type='checkbox'][name=...] (and [value=...] if needed).
                - Prefer stable attributes/scoping: name, id, value, data-*, aria-*, and clear form/field containers.
                - Avoid brittle patterns: :nth-*, positional indices, transient-state selectors ([aria-expanded], inline styles), overly generic roots (html body ...).
                - Do NOT include iframe nodes in the selector string.

                GOOD SELECTORS (examples)
                input[type='checkbox'][name='terms']
                input#subscribe_newsletter
                form#application-form input[type='checkbox'][name='gdpr_consent']
                form[aria-label='Job application'] input[type='checkbox'][data-qa='privacy_policy']
                #apply-modal.open form#apply-modal-form input[type='checkbox'][name='share_with_recruiters']
                form.ApplicationForm input[type='checkbox'][name='skills'][value='python']

                BAD SELECTORS (and why)
                label:has-text('I agree')                         # label, not the input
                input[type='radio'][name='agree']                 # wrong control type
                [style*='display:none'] input[type='checkbox']    # hidden
                iframe#grnhse_iframe input[type='checkbox'][name='x']  # crosses iframe in selector
                html body input[type='checkbox']                  # overly generic
                input[type='checkbox']:nth-of-type(3)             # positional, brittle
                div.toggle                                        # custom UI, not the form input

                OUTPUT SCHEMA (STRICT JSON ARRAY)
                Return ONLY a JSON array of objects. Each object MUST have:
                - "field_selector": minimal, stable CSS selector for the checkbox input
                - "field_name": the name attribute or best-effort visible label (from HTML)
                - "field_type": "checkbox"
                - "field_is_visible": true only if visible and interactive now
                - "field_in_form": true only if inside the application form (or active modal form)
                - "field_value": true or false  # from preferences-only logic above (use preferences to determine the value)
                    Example:
                        preferences:
                            - "accept_privacy_policy": true
                            - "gdpr_consent": false
                        output:
                            Checkbox 1
                            checkbox label: "I agree to the privacy policy" (from HTML)
                            - "field_selector": "input[type='checkbox'][name='accept_privacy_policy']"
                            - "field_name": "accept_privacy_policy"
                            - "field_type": "checkbox"
                            - "field_is_visible": true
                            - "field_in_form": true
                            - "field_value": true
                            
                            Checkbox 2
                            checkbox label: "I consent to the GDPR" (from HTML)
                            - "field_selector": "input[type='checkbox'][name='gdpr_consent']"
                            - "field_name": "gdpr_consent"
                            - "field_type": "checkbox"
                            - "field_is_visible": true
                            - "field_in_form": true
                            - "field_value": false

                ORDERING
                - Preserve top-to-bottom DOM order within the active scope.

                VALIDATION & SELF-CHECK (DO NOT OUTPUT)
                - Each selector resolves to exactly one input[type='checkbox'].
                - No item is hidden/disabled/off-screen or outside the active application scope.
                - No labels/wrappers/custom UI in place of the checkbox input.
                - No iframe traversal in selectors.
                - If no qualifying checkboxes, output [].

                FINAL OUTPUT REQUIREMENT
                - Output ONLY the JSON array per the schema above. No extra text.
                """

            
            all_fields = generate_model(
                "You are an expert at analyzing HTML and identifying form inputs. Return only valid JSON with CSS selectors.",
                model_object_type=CheckboxApplicationFieldResponse,
                system_prompt=system_prompt,
                model="gpt-5-mini"
            ).fields
            
            all_fields = [field for field in all_fields if field.field_is_visible and field.field_in_form]
            
            print(all_fields)
            
            context_str = "iframe" if frame else "page"
            # Parse the JSON response
            checkbox_fields: List[CheckboxApplicationField] = [field for field in all_fields if field.field_type == FieldType.CHECKBOX]
            verified_checkbox_fields = self.verify_checkbox_fields(checkbox_fields, frame, iframe_context, spinner)
            dprint(f"✅ gpt-5-mini found {len(checkbox_fields)} checkbox fields in {context_str}")
            
            return verified_checkbox_fields
            
        except Exception as e:
            print(f"❌ Error in find_all_checkbox_fields: {e}")
            return []

    def verify_select_fields(self, select_fields: List[SelectApplicationField], frame=None, iframe_context=None, spinner:Yaspin=None, best_of: int = 3) -> List[SelectApplicationField]:
        """
        Verify the select fields are valid using best_of parameter for consensus
        """
        try:
            if frame:
                page_content = frame.content()
            else:
                page_content = self.page.content()
            
            if best_of <= 1:
                # Single run - use original logic
                system_prompt = f"""
                    ROLE
                    You are a senior HTML/DOM form analyst. Verify a proposed list of SELECT dropdown fields against the actual page. If every item is valid, return the input list UNCHANGED. If any item fails, discard the list and RECOMPUTE the correct list from the HTML/screenshot. Output ONLY the final JSON array.

                    INPUTS
                    - html: {page_content}
                    - proposed_selector: {json.dumps([self._serialize_field_for_json(field) for field in select_fields])}
                    - NOTE: {"The proposed selectors list is empty - you MUST find all SELECT dropdown fields from scratch" if not select_fields else "The proposed selectors exist and should be validated"}

                    TARGET SCHEMA (ApplicationField)
                    Each object MUST have:
                    - "field_selector": minimal, stable CSS selector for the dropdown control
                    - "field_name": name attribute or best-effort visible label
                    - "field_type": "select"
                    - "field_is_visible": true only if visible & interactive now
                    - "field_in_form": true only if inside the application form (or active modal form)
                    - "field_options": array of visible options if available, else []

                    WHAT QUALIFIES AS A DROPDOWN
                    - Native <select> with <option> children  → selector targets the <select> element itself.
                    - React-Select–style dropdowns           → selector targets the clickable control container (e.g., div.select__control), NOT menus/options.

                    ACTIVE SCOPE & VISIBILITY
                    - If a visible modal/dialog is open (role="dialog" or [aria-modal="true"]), TREAT IT AS THE ONLY ACTIVE SCOPE; ignore background fields.
                    - Include ONLY elements that are visible & interactive (not display:none, visibility:hidden, opacity:0, off-screen/zero-size, [hidden], [aria-hidden="true"], [disabled], or aria-disabled="true"]).
                    - Include ONLY elements that belong to the job-application form (a relevant <form> ancestor).
                    - Do NOT cross iframe boundaries in selectors. If a field is inside an iframe, write the selector as if already inside that iframe DOM (no iframe prefix).

                    FORBIDDEN (NEVER VALID)
                    - Selectors that include iframe nodes (e.g., iframe#... ...)
                    - Menus/transient nodes: .select__menu, [role='listbox'], .select__option
                    - Labels/spans instead of the control: label[for=...], span.field-label
                    - Options themselves: option[value=...]
                    - Overly generic roots: html body select, html body div.select__control
                    - Positional/transient-state selectors: :nth-*, [aria-expanded=...], inline style predicates, off-screen portals
                    - Hidden/disabled elements
                    - Pseudo-dropdowns that are really text inputs/autocomplete without a stable clickable control (e.g., bare input[role='combobox'] with no stable select__control container)

                    VALIDATION CHECKLIST (APPLY TO EACH proposed item)
                    1) Selector resolves to EXACTLY ONE element in the active scope.
                    2) Element is a valid dropdown control:
                    - Native: tagName == 'SELECT'
                    - React-Select: clickable control container (e.g., .select__control), NOT menu/option/listbox
                    3) Element is visible & interactive now; not hidden/disabled/off-screen/future-step.
                    4) Element is inside the job-application form (or active modal form).
                    5) No iframe traversal in the selector.
                    6) field_type == "select"; field_is_visible == true; field_in_form == true.
                    7) field_selector is minimal & stable (prefer id/name/data-*; avoid positional/state-based hacks).
                    8) If native options are readily available, field_options lists them; otherwise [].

                    DECISION RULE
                    - If ALL proposed items pass EVERY validation and no qualifying dropdowns are missing, RETURN THE proposed_fields_json UNCHANGED (identical order and text).
                    - Otherwise, DISCARD proposed_fields_json and RECOMPUTE:
                    a) Re-detect all qualifying dropdowns from html/screenshot per the rules above.
                    b) Build minimal, stable selectors (native <select> or React-Select control container).
                    c) Order by top-to-bottom DOM order within the active scope.
                    d) Exclude anything forbidden.

                    OUTPUT
                    - Output ONLY the final JSON array of ApplicationField objects. NO extra text.
                    - If nothing qualifies, output [].

                    REASONING PROTOCOL (DO NOT REVEAL)
                    - Silently validate proposed_fields_json against the checklist.
                    - If any failure or missing field is detected, redo detection from scratch and produce a corrected list.
                    - Self-check the final array before output: uniqueness, visibility, scope, correct control type, forbidden patterns absent, no iframe traversal.
                    """
                
                all_fields = generate_model(
                    "You are an expert at analyzing HTML and identifying form inputs. Return only valid JSON with CSS selectors.",
                    model_object_type=SelectApplicationFieldResponse,
                    system_prompt=system_prompt,
                    model="gpt-5-mini"
                ).fields
                
                all_fields = [field for field in all_fields if field.field_is_visible and field.field_in_form]
                if debug_mode:
                    spinner.write(f"✅ verified {len(all_fields)} fields")
                
                # Filter out fields that are not visible
                context_str = "iframe" if frame else "page"
                
                # Parse the JSON response
                select_fields: List[SelectApplicationField] = [field for field in all_fields if field.field_type == FieldType.SELECT]
                
                context_str = "iframe" if frame else "page"
                if debug_mode:
                    spinner.write(f"✅ gpt-5-mini found {len(select_fields)} select fields in {context_str}")
                
                return select_fields
            
            # Multiple runs for consensus
            results = []
            for i in range(best_of):
                try:
                    if spinner:
                        spinner.write(f"🔍 Run {i+1}/{best_of} for select fields verification...")
                    else:
                        print(f"🔍 Run {i+1}/{best_of} for select fields verification...")
                    
                    system_prompt = f"""
                        ROLE
                        You are a senior HTML/DOM form analyst. Verify a proposed list of SELECT dropdown fields against the actual page. If every item is valid, return the input list UNCHANGED. If any item fails, discard the list and RECOMPUTE the correct list from the HTML/screenshot. Output ONLY the final JSON array.

                        INPUTS
                        - html: {page_content}
                        - proposed_selector: {json.dumps([self._serialize_field_for_json(field) for field in select_fields])}
                        - NOTE: {"The proposed selectors list is empty - you MUST find all SELECT dropdown fields from scratch" if not select_fields else "The proposed selectors exist and should be validated"}

                        TARGET SCHEMA (ApplicationField)
                        Each object MUST have:
                        - "field_selector": minimal, stable CSS selector for the dropdown control
                        - "field_name": name attribute or best-effort visible label
                        - "field_type": "select"
                        - "field_is_visible": true only if visible & interactive now
                        - "field_in_form": true only if inside the application form (or active modal form)
                        - "field_options": array of visible options if available, else []

                        WHAT QUALIFIES AS A DROPDOWN
                        - Native <select> with <option> children  → selector targets the <select> element itself.
                        - React-Select–style dropdowns           → selector targets the clickable control container (e.g., div.select__control), NOT menus/options.

                        ACTIVE SCOPE & VISIBILITY
                        - If a visible modal/dialog is open (role="dialog" or [aria-modal="true"]), TREAT IT AS THE ONLY ACTIVE SCOPE; ignore background fields.
                        - Include ONLY elements that are visible & interactive (not display:none, visibility:hidden, opacity:0, off-screen/zero-size, [hidden], [aria-hidden="true"], [disabled], or aria-disabled="true"]).
                        - Include ONLY elements that belong to the job-application form (a relevant <form> ancestor).
                        - Do NOT cross iframe boundaries in selectors. If a field is inside an iframe, write the selector as if already inside that iframe DOM (no iframe prefix).

                        FORBIDDEN (NEVER VALID)
                        - Selectors that include iframe nodes (e.g., iframe#... ...)
                        - Menus/transient nodes: .select__menu, [role='listbox'], .select__option
                        - Labels/spans instead of the control: label[for=...], span.field-label
                        - Options themselves: option[value=...]
                        - Overly generic roots: html body select, html body div.select__control
                        - Positional/transient-state selectors: :nth-*, [aria-expanded=...], inline style predicates, off-screen portals
                        - Hidden/disabled elements
                        - Pseudo-dropdowns that are really text inputs/autocomplete without a stable clickable control (e.g., bare input[role='combobox'] with no stable select__control container)

                        VALIDATION CHECKLIST (APPLY TO EACH proposed item)
                        1) Selector resolves to EXACTLY ONE element in the active scope.
                        2) Element is a valid dropdown control:
                        - Native: tagName == 'SELECT'
                        - React-Select: clickable control container (e.g., .select__control), NOT menu/option/listbox
                        3) Element is visible & interactive now; not hidden/disabled/off-screen/future-step.
                        4) Element is inside the job-application form (or active modal form).
                        5) No iframe traversal in the selector.
                        6) field_type == "select"; field_is_visible == true; field_in_form == true.
                        7) field_selector is minimal & stable (prefer id/name/data-*; avoid positional/state-based hacks).
                        8) If native options are readily available, field_options lists them; otherwise [].

                        DECISION RULE
                        - If ALL proposed items pass EVERY validation and no qualifying dropdowns are missing, RETURN THE proposed_fields_json UNCHANGED (identical order and text).
                        - Otherwise, DISCARD proposed_fields_json and RECOMPUTE:
                        a) Re-detect all qualifying dropdowns from html/screenshot per the rules above.
                        b) Build minimal, stable selectors (native <select> or React-Select control container).
                        c) Order by top-to-bottom DOM order within the active scope.
                        d) Exclude anything forbidden.

                        OUTPUT
                        - Output ONLY the final JSON array of ApplicationField objects. NO extra text.
                        - If nothing qualifies, output [].

                        REASONING PROTOCOL (DO NOT REVEAL)
                        - Silently validate proposed_fields_json against the checklist.
                        - If any failure or missing field is detected, redo detection from scratch and produce a corrected list.
                        - Self-check the final array before output: uniqueness, visibility, scope, correct control type, forbidden patterns absent, no iframe traversal.
                        """
                    
                    result = generate_model(
                        "You are an expert at analyzing HTML and identifying form inputs. Return only valid JSON with CSS selectors.",
                        model_object_type=SelectApplicationFieldResponse,
                        system_prompt=system_prompt,
                        model="gpt-5-mini"
                    ).fields
                    
                    if result:
                        results.append(result)
                        if spinner:
                            spinner.write(f"🔍 Run {i+1} found {len(result)} fields")
                        else:
                            print(f"🔍 Run {i+1} found {len(result)} fields")
                    else:
                        if spinner:
                            spinner.write(f"🔍 Run {i+1} returned empty result")
                        else:
                            print(f"🔍 Run {i+1} returned empty result")
                        
                except Exception as e:
                    if spinner:
                        spinner.write(f"❌ Error in run {i+1}: {e}")
                    else:
                        print(f"❌ Error in run {i+1}: {e}")
                    continue
            
            if not results:
                if spinner:
                    spinner.write(f"❌ All {best_of} runs failed, returning original fields")
                else:
                    print(f"❌ All {best_of} runs failed, returning original fields")
                return select_fields
            
            # Find most common result by comparing field selectors
            from collections import Counter
            selector_counts = Counter()
            
            for result in results:
                for field in result:
                    if field.field_is_visible and field.field_in_form and field.field_type == FieldType.SELECT:
                        selector_counts[field.field_selector] += 1
            
            # Get the most common selectors
            most_common_selectors = selector_counts.most_common()
            
            if spinner:
                spinner.write(f"🔍 Consensus results: {dict(selector_counts)}")
            else:
                print(f"🔍 Consensus results: {dict(selector_counts)}")
            
            # If we have consensus on most selectors, use the most common result
            if most_common_selectors and most_common_selectors[0][1] > 1:
                # Find the result with the most common selectors
                best_result = max(results, key=lambda r: sum(selector_counts.get(f.field_selector, 0) for f in r if f.field_is_visible and f.field_in_form and f.field_type == FieldType.SELECT))
                
                best_fields = [field for field in best_result if field.field_is_visible and field.field_in_form and field.field_type == FieldType.SELECT]
                
                if spinner:
                    spinner.write(f"🔍 Using consensus result with {len(best_fields)} fields")
                else:
                    print(f"🔍 Using consensus result with {len(best_fields)} fields")
                
                return best_fields
            else:
                if spinner:
                    spinner.write(f"🔍 No clear consensus, returning last result with {len(results[-1]) if results else 0} fields")
                else:
                    print(f"🔍 No clear consensus, returning last result with {len(results[-1]) if results else 0} fields")
                return results[-1] if results else select_fields
        except Exception as e:
            print(f"❌ Error in verify_select_fields: {e}")
            return []

    def find_all_select_fields(self, frame=None, iframe_context=None, spinner:Yaspin=None) -> List[SelectApplicationField]:
        """
        Find all select fields in page or iframe context
        
        Args:
            frame: Optional iframe frame context. If None, uses main page.
        """
        try:
            # Get page content from frame or main page
            if frame:
                page_content = frame.content()
            else:
                page_content = self.page.content()

            # Use gpt-5-mini to analyze the page and identify form fields
            system_prompt = f"""
            ROLE
            You are a senior HTML/DOM form analyst. Your job is to return ONLY SELECT-style dropdown fields from a job application experience as strict JSON usable by Python Playwright.

            INPUTS
            - preferences: {self.preferences}
            - html: {page_content}

            REASONING PROTOCOL (DO NOT REVEAL)
            - First, silently plan: identify the active scope (modal vs page), enumerate candidate elements, filter by visibility/eligibility, then choose minimal stable selectors.
            - Perform a quick self-check against the rules and forbidden patterns.
            - OUTPUT ONLY the final JSON array. No commentary, no thoughts, no explanations.

            OBJECTIVE
            Extract every visible, interactive dropdown that lets a user choose one or multiple options.
            Supported types:
            1) Native <select> with <option>
            2) React-Select–style components whose clickable control is a stable container (commonly a div with class*="select__control").
            Ignore pseudo-dropdowns that are actually text inputs/autocomplete widgets without a stable clickable control.

            SCOPE & VISIBILITY RULES
            - Only fields inside the job-application form (i.e., a <form> ancestor clearly tied to applying).
            - Only currently visible and interactive (not hidden, disabled, off-screen, or zero-size).
            - If a visible modal/dialog is open (role="dialog" or [aria-modal="true"]), TREAT THE MODAL AS THE ACTIVE SCOPE and ignore background fields.
            - Ignore future-step or collapsed sections not currently expanded/active.
            - Never cross iframes. Do not return selectors that include iframe boundaries.

            NATIVE <select> RULES
            - Target the <select> element directly with a minimal, stable CSS selector.
            - Prefer stable attributes and scoping: name, id, data-*, or a clear form/field container.
            GOOD (native)
            select[name='country']
            select#job_type
            form#application-form select[name='department']
            form[action='/apply'] select[name='education_level']
            form[aria-label='Job application'] select[data-qa='availability']
            form#apply-modal-form select#notice_period
            form.ApplicationForm select[name='work_authorization']
            form#application-form fieldset.location select[name='city']
            form#application-form section.step-current select[name='seniority']
            form#application-form select[name='salary_currency']
            form#application-form select[multiple][name='skills']
            form#application-form .benefits select[name='benefit_selection']
            BAD (native) — and why
            select:nth-of-type(2)                          # positional, brittle
            form#application-form select[style*='display:none']  # hidden
            select[disabled]                               # not interactive
            .sidebar select[name='profile_location']       # outside application form
            iframe#grnhse_iframe select[name='country']    # crosses iframe
            html body select                               # generic/unstable
            label[for='country']                           # not the control
            option[value='gb']                             # not the control
            select[aria-hidden='true']                     # hidden
            form#application-form [role='listbox']         # popup semantics, not the control

            REACT-SELECT (OR SIMILAR) RULES
            - Target the clickable control container (e.g., .select__control), NOT the menu list or transient option nodes.
            - Scope to the application form/field container to disambiguate multiples.
            GOOD (react-select)
            div.select__control
            div[class*='select__control']
            form#job-application div.select__control
            form#application-form .experience .select__control
            form#application-form [data-testid='exp-select'].select__control
            #apply-modal.open form#apply-modal-form .select__control[data-testid='job-type']
            form#application-form .job-type .select-shell .select__control
            form.ApplicationForm [data-field='department'] .select__control
            form#application-form .rs-container > .select__control
            form#application-form .field[data-field='employment_type'] .select__control
            form#application-form section.step-current .select__control
            form#application-form [aria-label='Country'] ~ .select-shell .select__control   # only if structure is stable and this is the clickable control
            BAD (react-select) — and why
            div.select__menu                           # transient popup
            div[role='listbox']                        # popup list, not the control
            .select__option                            # transient option
            #react-select-3-input                       # dynamic IDs
            html body div.select__control              # generic, no form scope
            .select__control[aria-expanded='true']     # state-dependent
            .select__control:has(+ .select__menu)      # portalized menus break this
            div.select__control[aria-labelledby='question_322...']  # ARIA coupling, brittle
            #question_322...-label ~ .select-shell .select__control # label adjacency, brittle
            body iframe iframe div.select__control     # crosses documents
            div.select__menu[style*='left:-9999px']    # off-screen portal

            MODAL/DIALOG SCOPING
            - If a visible modal/dialog is open, only return fields from the modal's form; ignore the background.
            GOOD (modal)
            #apply-modal.open form#apply-modal-form select[name='notice_period']
            #apply-modal.open form#apply-modal-form .select__control[data-testid='job-type']
            div[role='dialog'][aria-modal='true'] form select[name='availability']
            BAD (modal)
            form#application-form select[name='notice_period']   # background while modal is active
            [aria-hidden='true'] select                          # hidden behind modal

            HIDDEN/FUTURE-STEP EXCLUSIONS
            - Exclude nodes with display:none, visibility:hidden, hidden attribute, aria-hidden='true', off-screen positioning, or zero size.
            - Exclude collapsed accordions/tabs unless visibly active/expanded (e.g., class contains 'open' or 'active').
            BAD
            form#application-form .future-step select[name='seniority']         # future step
            form#application-form .tab-panel:not(.active) select[name='x']      # inactive tab

            FORBIDDEN SELECTORS (NEVER USE)
            - Any selector that includes iframes/cross-document traversal
            - Menus/options/transient nodes: .select__menu, [role='listbox'], .select__option
            - Labels/spans instead of the control (e.g., label[for=...], span.field-label)
            - Overly generic paths (e.g., html body div, html body select)
            - Positional or transient-state selectors (:nth-*, [aria-expanded='true'], etc.)
            - Hidden/off-screen/disabled nodes ([hidden], [disabled], [style*='left:-9999px'], [aria-hidden='true'])

            OUTPUT SCHEMA (STRICT JSON ARRAY)
            For each dropdown, return an object with exactly:
            - "field_selector": minimal, stable CSS selector for the control
            - "field_name": the name attribute or best-effort visible label (from HTML/screenshot)
            - "field_type": "select"
            - "field_is_visible": true if and only if interactive and visible now
            - "field_in_form": true if and only if inside the application form (or active modal form)
            - "field_value": "" (empty string)
            - "field_options": array of visible options if available, else []

            ORDERING
            - Preserve top-to-bottom DOM order within the active scope.

            VALIDATION & SELF-CHECK (DO NOT OUTPUT)
            - Each selector resolves to exactly one control element of the correct type.
            - No item is hidden/disabled/off-screen or outside the active application scope.
            - No item targets menus/options/labels/iframed content.
            - If no qualifying dropdowns, output [].

            FINAL OUTPUT REQUIREMENT
            - Output ONLY the JSON array per the schema above. No extra text.
            """
            all_fields = generate_model(
                "You are an expert at analyzing HTML and identifying form inputs. Return only valid JSON with CSS selectors.",
                model_object_type=SelectApplicationFieldResponse,
                system_prompt=system_prompt,
                model="gpt-5-mini"
            ).fields
            
            all_fields = [field for field in all_fields if field.field_is_visible and field.field_in_form]
            
            # Filter out fields that are not visible
            context_str = "iframe" if frame else "page"
            
            # Parse the JSON response
            select_fields: List[SelectApplicationField] = [field for field in all_fields if field.field_type == FieldType.SELECT]
            
            verified_select_fields = self.verify_select_fields(select_fields, frame, iframe_context, spinner)
            
            context_str = "iframe" if frame else "page"
            dprint(f"✅ gpt-5-mini found {len(verified_select_fields)} select fields in {context_str}")
            
            return verified_select_fields
            
        except Exception as e:
            print(f"❌ Error in find_and_handle_all_select_fields: {e}")
            traceback.print_exc()
            return []

    def verify_upload_file_button(self, upload_field: UploadApplicationField, frame=None, iframe_context=None, spinner:Yaspin=None, best_of: int = 3) -> UploadApplicationField:
        """
        Verify the upload file button is valid using best_of parameter for consensus
        """
        try:
            if frame:
                page_content = frame.content()
            else:
                page_content = self.page.content()
            
            if best_of <= 1:
                # Single run - use original logic
                system_prompt = f"""
                ROLE
                You are a senior HTML/DOM analyst. Verify a proposed Playwright CSS selector for the file-upload trigger (resume/CV/cover letter). If it is fully valid, return it UNCHANGED. If it fails any check, IGNORE it and RECOMPUTE the correct selector from html/screenshot. Output EXACTLY ONE line: the selector string or an empty string.

                INPUTS
                - html: {page_content}
                - proposed_selector: {upload_field.field_selector}
                - NOTE: {"The proposed selector is empty - you MUST find a file upload button from scratch" if not upload_field.field_selector else "The proposed selector exists and should be validated"}

                TARGET
                A single selector for a user-visible control that OPENS the OS file picker:
                - Element type must be <button>, <a>, or <div acting as a button> (e.g., role='button', click handler).
                - NEVER target <input> (even input[type='file'] is forbidden).

                ACTIVE SCOPE & VISIBILITY
                - If a visible modal/dialog exists (role="dialog" or [aria-modal="true"]), it is the ONLY active scope; ignore background.
                - Include only elements that are visible & interactive now (not display:none, visibility:hidden, opacity:0, off-screen/zero-size, [hidden], [aria-hidden="true"], [disabled], aria-disabled="true"]).
                - Do NOT cross iframe boundaries in the selector. If the control is inside an iframe, write the selector as if already inside that iframe's DOM (no iframe prefix).

                FORBIDDEN (NEVER VALID)
                - Any selector that targets <input> (including input[type='file'])
                - Selectors that include iframe nodes or cross-document paths
                - Overly generic roots (e.g., html body button, body a)
                - Positional/transient-state hacks: :nth-*, [aria-expanded], inline style predicates
                - Hidden/disabled elements
                - Non-upload actions: remove/replace/delete/download/view/preview/cancel/close/submit/apply/next/continue/save/settings/preferences

                UPLOAD SEMANTIC SIGNALS (positive examples; case-insensitive)
                - Text/label/aria/title includes: upload, attach, add file, choose file, browse, select file, import, add resume, upload CV, attach cover letter
                - Iconography/aria consistent with upload (e.g., button:has(svg[aria-label*='upload']))

                VALIDATION CHECKLIST (APPLY TO proposed_selector)
                1) Resolves to EXACTLY ONE element in active scope.
                2) Element is <button>, <a>, or <div> with button-like semantics (role='button' or clearly clickable).
                3) Element is visible & enabled now.
                4) Selector has NO iframe prefixes.
                5) Selector does NOT target an <input>.
                6) Element's accessible name/text/attributes suggest file upload (positive signals) and do not match forbidden actions.
                7) Selector is reasonably stable (id/data-*/aria/clear container + :has-text or attribute). Not overly generic or positional.

                DECISION RULE
                - If ALL checklist items pass AND no stronger/more primary upload trigger is present in the same scope, RETURN proposed_selector UNCHANGED.
                - Otherwise, RECOMPUTE:
                a) Enumerate candidate controls per upload semantic signals among visible <button>/<a>/<div role='button'>.
                b) Exclude forbidden/negative actions and wrappers/containers that are not the actual clickable trigger.
                c) Prefer the primary trigger (e.g., "Upload/Attach/Choose file", "Upload resume/CV"), then build the most stable selector:
                    - Prefer #id, [data-*], [aria-label], [title], [data-testid], [data-qa].
                    - Otherwise: nearest stable uploader container + button with :has-text(/upload|attach|choose|browse|select file|resume|cv/i).
                d) Ensure uniqueness and visibility; no iframe prefixes; not an <input>.
                e) If multiple equally valid candidates remain, pick the first in top-to-bottom DOM order.

                OUTPUT
                - Output EXACTLY ONE line containing ONLY the selector string (no quotes, no extra text).
                - If no qualifying control exists in the active scope, output an empty string.

                REASONING PROTOCOL (DO NOT REVEAL)
                - Validate proposed_selector with the checklist.
                - If invalid/ambiguous, recompute per the procedure.
                - Final self-check before output: uniqueness, visibility, correct element type, upload semantics, no forbidden patterns.
                """

                
                upload_field = generate_model(
                    "You are an expert at analyzing HTML and identifying form inputs. Return only valid JSON with CSS selectors.",
                    UploadApplicationField,
                    system_prompt=system_prompt,
                    model="gpt-5-mini"
                )
                
                if debug_mode:
                    spinner.write(f"✅ verified upload file button")
                
                return upload_field
            
            # Multiple runs for consensus
            results = []
            for i in range(best_of):
                try:
                    if spinner:
                        spinner.write(f"🔍 Run {i+1}/{best_of} for upload file button verification...")
                    else:
                        print(f"🔍 Run {i+1}/{best_of} for upload file button verification...")
                    
                    system_prompt = f"""
                    ROLE
                    You are a senior HTML/DOM analyst. Verify a proposed Playwright CSS selector for the file-upload trigger (resume/CV/cover letter). If it is fully valid, return it UNCHANGED. If it fails any check, IGNORE it and RECOMPUTE the correct selector from html/screenshot. Output EXACTLY ONE line: the selector string or an empty string.

                    INPUTS
                    - html: {page_content}
                    - proposed_selector: {upload_field.field_selector}
                    - NOTE: {"The proposed selector is empty - you MUST find a file upload button from scratch" if not upload_field.field_selector else "The proposed selector exists and should be validated"}

                    TARGET
                    A single selector for a user-visible control that OPENS the OS file picker:
                    - Element type must be <button>, <a>, or <div acting as a button> (e.g., role='button', click handler).
                    - NEVER target <input> (even input[type='file'] is forbidden).

                    ACTIVE SCOPE & VISIBILITY
                    - If a visible modal/dialog exists (role="dialog" or [aria-modal="true"]), it is the ONLY active scope; ignore background.
                    - Include only elements that are visible & interactive now (not display:none, visibility:hidden, opacity:0, off-screen/zero-size, [hidden], [aria-hidden="true"], [disabled], aria-disabled="true"]).
                    - Do NOT cross iframe boundaries in the selector. If the control is inside an iframe, write the selector as if already inside that iframe's DOM (no iframe prefix).

                    FORBIDDEN (NEVER VALID)
                    - Any selector that targets <input> (including input[type='file'])
                    - Selectors that include iframe nodes or cross-document paths
                    - Overly generic roots (e.g., html body button, body a)
                    - Positional/transient-state hacks: :nth-*, [aria-expanded], inline style predicates
                    - Hidden/disabled elements
                    - Non-upload actions: remove/replace/delete/download/view/preview/cancel/close/submit/apply/next/continue/save/settings/preferences

                    UPLOAD SEMANTIC SIGNALS (positive examples; case-insensitive)
                    - Text/label/aria/title includes: upload, attach, add file, choose file, browse, select file, import, add resume, upload CV, attach cover letter
                    - Iconography/aria consistent with upload (e.g., button:has(svg[aria-label*='upload']))

                    VALIDATION CHECKLIST (APPLY TO proposed_selector)
                    1) Resolves to EXACTLY ONE element in active scope.
                    2) Element is <button>, <a>, or <div> with button-like semantics (role='button' or clearly clickable).
                    3) Element is visible & enabled now.
                    4) Selector has NO iframe prefixes.
                    5) Selector does NOT target an <input>.
                    6) Element's accessible name/text/attributes suggest file upload (positive signals) and do not match forbidden actions.
                    7) Selector is reasonably stable (id/data-*/aria/clear container + :has-text or attribute). Not overly generic or positional.

                    DECISION RULE
                    - If ALL checklist items pass AND no stronger/more primary upload trigger is present in the same scope, RETURN proposed_selector UNCHANGED.
                    - Otherwise, RECOMPUTE:
                    a) Enumerate candidate controls per upload semantic signals among visible <button>/<a>/<div role='button'>.
                    b) Exclude forbidden/negative actions and wrappers/containers that are not the actual clickable trigger.
                    c) Prefer the primary trigger (e.g., "Upload/Attach/Choose file", "Upload resume/CV"), then build the most stable selector:
                        - Prefer #id, [data-*], [aria-label], [title], [data-testid], [data-qa].
                        - Otherwise: nearest stable uploader container + button with :has-text(/upload|attach|choose|browse|select file|resume|cv/i).
                    d) Ensure uniqueness and visibility; no iframe prefixes; not an <input>.
                    e) If multiple equally valid candidates remain, pick the first in top-to-bottom DOM order.

                    OUTPUT
                    - Output EXACTLY ONE line containing ONLY the selector string (no quotes, no extra text).
                    - If no qualifying control exists in the active scope, output an empty string.

                    REASONING PROTOCOL (DO NOT REVEAL)
                    - Validate proposed_selector with the checklist.
                    - If invalid/ambiguous, recompute per the procedure.
                    - Final self-check before output: uniqueness, visibility, correct element type, upload semantics, no forbidden patterns.
                    """

                    result = generate_model(
                        "You are an expert at analyzing HTML and identifying form inputs. Return only valid JSON with CSS selectors.",
                        UploadApplicationField,
                        system_prompt=system_prompt,
                        model="gpt-5-mini"
                    )
                    
                    if result:
                        results.append(result)
                        if spinner:
                            spinner.write(f"🔍 Run {i+1} found upload button")
                        else:
                            print(f"🔍 Run {i+1} found upload button")
                    else:
                        if spinner:
                            spinner.write(f"🔍 Run {i+1} returned empty result")
                        else:
                            print(f"🔍 Run {i+1} returned empty result")
                        
                except Exception as e:
                    if spinner:
                        spinner.write(f"❌ Error in run {i+1}: {e}")
                    else:
                        print(f"❌ Error in run {i+1}: {e}")
                    continue
            
            if not results:
                if spinner:
                    spinner.write(f"❌ All {best_of} runs failed, returning original field")
                else:
                    print(f"❌ All {best_of} runs failed, returning original field")
                return upload_field
            
            # Find most common result by comparing field selectors
            from collections import Counter
            selector_counts = Counter()
            
            for result in results:
                if result and hasattr(result, 'field_selector'):
                    selector_counts[result.field_selector] += 1
            
            # Get the most common selectors
            most_common_selectors = selector_counts.most_common()
            
            if spinner:
                spinner.write(f"🔍 Consensus results: {dict(selector_counts)}")
            else:
                print(f"🔍 Consensus results: {dict(selector_counts)}")
            
            # If we have consensus on the selector, use the most common result
            if most_common_selectors and most_common_selectors[0][1] > 1:
                # Find the result with the most common selector
                best_result = max(results, key=lambda r: selector_counts.get(r.field_selector, 0) if r and hasattr(r, 'field_selector') else 0)
                
                if spinner:
                    spinner.write(f"🔍 Using consensus result: {best_result.field_selector}")
                else:
                    print(f"🔍 Using consensus result: {best_result.field_selector}")
                
                return best_result
            else:
                if spinner:
                    spinner.write(f"🔍 No clear consensus, returning last result")
                else:
                    print(f"🔍 No clear consensus, returning last result")
                return results[-1] if results else upload_field
        except Exception as e:
            print(f"❌ Error in verify_upload_file_button: {e}")
            return None

    def find_upload_file_button(self, frame=None, iframe_context=None, spinner:Yaspin=None) -> UploadApplicationField:
        """
        Find upload file button in page or iframe context
        
        Args:
            frame: Optional iframe frame context. If None, uses main page.
        """
        try:
            # Get page content from frame or main page
            if frame:
                page_content = frame.content()
            else:
                page_content = self.page.content()
            
            # Use gpt-5-mini to analyze the page and identify form fields
            system_prompt = f"""
            ROLE
            You are a senior front-end/DOM analyst. Return EXACTLY ONE Playwright-compatible selector for the user-visible control that opens a file picker to upload a resume/CV/cover letter. The control is a clickable <button>, <a>, or <div> acting as a button. Never return an <input> selector.

            INPUTS
            - preferences: {self.preferences}
            - html: {page_content}

            REASONING PROTOCOL (DO NOT REVEAL)
            1) Determine active scope: if a visible modal/dialog is open (role='dialog' or [aria-modal='true']), restrict to that; otherwise use the page.
            2) Enumerate candidate controls: visible, enabled <button>, <a>, or <div role='button'|button-like) whose text/label/title/aria suggests file upload (upload, attach, add file, choose/browse/select file, import resume/CV/cover letter).
            3) Exclude negatives: remove/replace/delete/download/view/preview/cancel/close/submit/apply/next/continue/save/settings/preferences.
            4) Prefer the primary upload trigger that opens the OS file chooser (not a submit button, not a wrapper/container, not a hidden proxy).
            5) Build a unique, stable selector (id/data-* preferred; otherwise scoped container + :has-text(...) or attribute match).
            6) Self-check: selector resolves to exactly one visible element in the active scope; no iframe prefixes; not an <input>.

            OBJECTIVE
            Return a robust selector that, when clicked, initiates the file selection flow for resume/CV/cover letter upload.

            SCOPE & VISIBILITY
            - Include only elements currently visible and interactive (exclude display:none, visibility:hidden, opacity:0, aria-hidden='true', off-screen/zero-size, disabled).
            - If a modal/dialog is visible, ignore background page controls.
            - Do NOT cross iframes in the selector. If the control lives inside an iframe, write the selector as if already inside that iframe's DOM (no iframe prefix).

            SELECTOR RULES (Playwright CSS)
            - Target the actual clickable control (<button>, <a>, or <div> with role='button'/onClick).
            - Prefer stable attributes: #id, [data-*], [aria-label], [title], [data-testid], [data-qa].
            - May use :has-text("...") or :has-text(/.../i) to disambiguate.
            - Avoid brittle patterns: :nth-*, overly generic roots (html body ...), transient state selectors ([aria-expanded], inline style predicates).

            ALLOWED EXAMPLES (do not output verbatim)
            button[aria-label='Upload resume']
            button#resumeUpload
            div.upload-btn[role='button']
            a[data-testid='fileUploader']
            div[class*='upload'][role='button']
            button[type='button'][data-action='upload']
            .resume-section button:not([disabled])
            button:has(svg[aria-label='Upload'])
            a[aria-label='Attach file']
            div[role='button'][data-qa='attachment']
            section.cv-uploader button:has-text(/upload|attach|choose|browse/i)

            FORBIDDEN EXAMPLES (and why)
            input[type='file']                               # input, not allowed
            iframe#grnhse_iframe button[type='file']         # crosses iframe boundary
            div                                              # hopelessly generic
            body button                                      # too broad
            button:nth-child(3)                              # positional/brittle
            div[aria-labelledby='upload-label-resume'] .button-container button  # wrapper, not the control
            .modal[style*='display:none'] button             # hidden
            form:first-of-type button[type='submit']         # submit, not upload
            a[href='#']                                      # generic anchor without semantics

            """

            
            upload_field = generate_model(
                "You are an expert at analyzing HTML and identifying form inputs. Return only valid JSON with CSS selectors.",
                UploadApplicationField,
                system_prompt=system_prompt,
                model="gpt-5-mini"
            )
            
            verified_upload_field = self.verify_upload_file_button(upload_field, frame, iframe_context, spinner)
            
            # Parse the JSON response
            return verified_upload_field
            
        except Exception as e:
            print(f"❌ Error in find_upload_file_button: {e}")
            return None

    def verify_checkbox_fields(self, checkbox_fields: List[CheckboxApplicationField], frame=None, iframe_context=None, spinner:Yaspin=None, best_of: int = 3) -> List[CheckboxApplicationField]:
        """
        Verify the checkbox fields are valid using best_of parameter for consensus
        """
        try:
            if frame:
                page_content = frame.content()
            else:
                page_content = self.page.content()
            
            if best_of <= 1:
                # Single run - use original logic
                system_prompt = f"""
                    ROLE
                    You are a senior HTML/DOM form analyst. Verify a proposed list of CHECKBOX fields against the actual page. If every item is valid, return the input list UNCHANGED. If any item fails, discard the list and RECOMPUTE the correct list from the HTML/screenshot. Output ONLY the final JSON array.

                    INPUTS
                    - preferences: {self.preferences}
                    - html: {page_content}
                    - proposed_checkboxes: {json.dumps([self._serialize_field_for_json(field) for field in checkbox_fields])}
                    - NOTE: {"The proposed checkboxes list is empty - you MUST find all CHECKBOX fields from scratch" if not checkbox_fields else "The proposed checkboxes exist and should be validated"}

                    TARGET SCHEMA (ApplicationField)
                    Each object MUST have:
                    - "field_selector": minimal, stable CSS selector for the checkbox control
                    - "field_name": name attribute or best-effort visible label
                    - "field_type": "checkbox"
                    - "field_is_visible": true only if visible & interactive now
                    - "field_in_form": true only if inside the application form (or active modal form)
                    - "field_checked": false (default unchecked state)

                    WHAT QUALIFIES AS A CHECKBOX
                    - Native <input type="checkbox"> elements
                    - Custom checkbox implementations with role="checkbox" and proper ARIA attributes
                    - Toggle switches and other binary choice controls that behave like checkboxes

                    ACTIVE SCOPE & VISIBILITY
                    - If a visible modal/dialog is open (role="dialog" or [aria-modal="true"]), TREAT IT AS THE ONLY ACTIVE SCOPE; ignore background fields.
                    - Include ONLY elements that are visible & interactive (not display:none, visibility:hidden, opacity:0, off-screen/zero-size, [hidden], [aria-hidden="true"], [disabled], or aria-disabled="true"]).
                    - Include ONLY elements that belong to the job-application form (a relevant <form> ancestor).
                    - Do NOT cross iframe boundaries in selectors. If a field is inside an iframe, write the selector as if already inside that iframe DOM (no iframe prefix).

                    FORBIDDEN (NEVER VALID)
                    - Selectors that include iframe nodes (e.g., iframe#... ...)
                    - Labels/spans instead of the control: label[for=...], span.field-label
                    - Radio buttons: input[type="radio"]
                    - Text inputs: input[type="text"], input[type="email"], etc.
                    - Overly generic roots: html body input, html body div
                    - Positional/transient-state selectors: :nth-*, [aria-expanded=...], inline style predicates, off-screen portals
                    - Hidden/disabled elements
                    - Non-checkbox controls: buttons, links, selects, textareas

                    VALIDATION CHECKLIST (APPLY TO EACH proposed item)
                    1) Selector resolves to EXACTLY ONE element in the active scope.
                    2) Element is a valid checkbox control:
                    - Native: tagName == 'INPUT' and type == 'checkbox'
                    - Custom: has role="checkbox" or equivalent checkbox semantics
                    3) Element is visible & interactive now; not hidden/disabled/off-screen/future-step.
                    4) Element is inside the job-application form (or active modal form).
                    5) No iframe traversal in the selector.
                    6) field_type == "checkbox"; field_is_visible == true; field_in_form == true.
                    7) field_selector is minimal & stable (prefer id/name/data-*; avoid positional/state-based hacks).

                    DECISION RULE
                    - If ALL proposed items pass EVERY validation and no qualifying checkboxes are missing, RETURN THE proposed_fields_json UNCHANGED (identical order and text).
                    - Otherwise, DISCARD proposed_fields_json and RECOMPUTE:
                    a) Re-detect all qualifying checkboxes from html/screenshot per the rules above.
                    b) Build minimal, stable selectors (native <input type="checkbox"> or custom checkbox with role="checkbox").
                    c) Order by top-to-bottom DOM order within the active scope.
                    d) Exclude anything forbidden.
                            
                OUTPUT SCHEMA (STRICT JSON ARRAY)
                Return ONLY a JSON array of objects. Each object MUST have:
                - "field_selector": minimal, stable CSS selector for the checkbox input
                - "field_name": the name attribute or best-effort visible label (from HTML)
                - "field_type": "checkbox"
                - "field_is_visible": true only if visible and interactive now
                - "field_in_form": true only if inside the application form (or active modal form)
                - "field_value": true or false  # from preferences-only logic above (use preferences to determine the value)
                    Example:
                        preferences:
                            - "accept_privacy_policy": true
                            - "gdpr_consent": false
                        output:
                            Checkbox 1
                            checkbox label: "I agree to the privacy policy" (from HTML)
                            - "field_selector": "input[type='checkbox'][name='accept_privacy_policy']"
                            - "field_name": "accept_privacy_policy"
                            - "field_type": "checkbox"
                            - "field_is_visible": true
                            - "field_in_form": true
                            - "field_value": true
                            
                            Checkbox 2
                            checkbox label: "I consent to the GDPR" (from HTML)
                            - "field_selector": "input[type='checkbox'][name='gdpr_consent']"
                            - "field_name": "gdpr_consent"
                            - "field_type": "checkbox"
                            - "field_is_visible": true
                            - "field_in_form": true
                            - "field_value": false

                    OUTPUT
                    - Output ONLY the final JSON array of ApplicationField objects. NO extra text.
                    - If nothing qualifies, output [].

                    REASONING PROTOCOL (DO NOT REVEAL)
                    - Silently validate proposed_fields_json against the checklist.
                    - If any failure or missing field is detected, redo detection from scratch and produce a corrected list.
                    - Self-check the final array before output: uniqueness, visibility, scope, correct control type, forbidden patterns absent, no iframe traversal.
                    """
                
                all_fields = generate_model(
                    "You are an expert at analyzing HTML and identifying form inputs. Return only valid JSON with CSS selectors.",
                    model_object_type=CheckboxApplicationFieldResponse,
                    system_prompt=system_prompt,
                    model="gpt-5-mini"
                ).fields
                
                all_fields = [field for field in all_fields if field.field_is_visible and field.field_in_form]
                if debug_mode:
                    spinner.write(f"✅ verified {len(all_fields)} fields")
                
                # Filter out fields that are not visible
                context_str = "iframe" if frame else "page"
                
                # Parse the JSON response
                checkbox_fields: List[CheckboxApplicationField] = [field for field in all_fields if field.field_type == FieldType.CHECKBOX]
                
                context_str = "iframe" if frame else "page"
                if debug_mode:
                    spinner.write(f"✅ gpt-5-mini found {len(checkbox_fields)} checkbox fields in {context_str}")
                
                return checkbox_fields
            
            # Multiple runs for consensus
            results = []
            for i in range(best_of):
                try:
                    if spinner:
                        spinner.write(f"🔍 Run {i+1}/{best_of} for checkbox fields verification...")
                    else:
                        print(f"🔍 Run {i+1}/{best_of} for checkbox fields verification...")
                    
                    system_prompt = f"""
                        ROLE
                        You are a senior HTML/DOM form analyst. Verify a proposed list of CHECKBOX fields against the actual page. If every item is valid, return the input list UNCHANGED. If any item fails, discard the list and RECOMPUTE the correct list from the HTML/screenshot. Output ONLY the final JSON array.

                        INPUTS
                        - html: {page_content}
                        - preferences: {self.preferences}
                        - proposed_selector: {json.dumps([self._serialize_field_for_json(field) for field in checkbox_fields])}

                        TARGET SCHEMA (ApplicationField)
                        Each object MUST have:
                        - "field_selector": minimal, stable CSS selector for the checkbox control
                        - "field_name": name attribute or best-effort visible label
                        - "field_type": "checkbox"
                        - "field_is_visible": true only if visible & interactive now
                        - "field_in_form": true only if inside the application form (or active modal form)
                        - "field_checked": false (default unchecked state)

                        WHAT QUALIFIES AS A CHECKBOX
                        - Native <input type="checkbox"> elements
                        - Custom checkbox implementations with role="checkbox" and proper ARIA attributes
                        - Toggle switches and other binary choice controls that behave like checkboxes

                        ACTIVE SCOPE & VISIBILITY
                        - If a visible modal/dialog is open (role="dialog" or [aria-modal="true"]), TREAT IT AS THE ONLY ACTIVE SCOPE; ignore background fields.
                        - Include ONLY elements that are visible & interactive (not display:none, visibility:hidden, opacity:0, off-screen/zero-size, [hidden], [aria-hidden="true"], [disabled], or aria-disabled="true"]).
                        - Include ONLY elements that belong to the job-application form (a relevant <form> ancestor).
                        - Do NOT cross iframe boundaries in selectors. If a field is inside an iframe, write the selector as if already inside that iframe DOM (no iframe prefix).

                        FORBIDDEN (NEVER VALID)
                        - Selectors that include iframe nodes (e.g., iframe#... ...)
                        - Labels/spans instead of the control: label[for=...], span.field-label
                        - Radio buttons: input[type="radio"]
                        - Text inputs: input[type="text"], input[type="email"], etc.
                        - Overly generic roots: html body input, html body div
                        - Positional/transient-state selectors: :nth-*, [aria-expanded=...], inline style predicates, off-screen portals
                        - Hidden/disabled elements
                        - Non-checkbox controls: buttons, links, selects, textareas

                        VALIDATION CHECKLIST (APPLY TO EACH proposed item)
                        1) Selector resolves to EXACTLY ONE element in the active scope.
                        2) Element is a valid checkbox control:
                        - Native: tagName == 'INPUT' and type == 'checkbox'
                        - Custom: has role="checkbox" or equivalent checkbox semantics
                        3) Element is visible & interactive now; not hidden/disabled/off-screen/future-step.
                        4) Element is inside the job-application form (or active modal form).
                        5) No iframe traversal in the selector.
                        6) field_type == "checkbox"; field_is_visible == true; field_in_form == true.
                        7) field_selector is minimal & stable (prefer id/name/data-*; avoid positional/state-based hacks).

                        DECISION RULE
                        - If ALL proposed items pass EVERY validation and no qualifying checkboxes are missing, RETURN THE proposed_fields_json UNCHANGED (identical order and text).
                        - Otherwise, DISCARD proposed_fields_json and RECOMPUTE:
                        a) Re-detect all qualifying checkboxes from html/screenshot per the rules above.
                        b) Build minimal, stable selectors (native <input type="checkbox"> or custom checkbox with role="checkbox").
                        c) Order by top-to-bottom DOM order within the active scope.
                        d) Exclude anything forbidden.

                        OUTPUT
                        - Output ONLY the final JSON array of ApplicationField objects. NO extra text.
                        - If nothing qualifies, output [].

                        REASONING PROTOCOL (DO NOT REVEAL)
                        - Silently validate proposed_fields_json against the checklist.
                        - If any failure or missing field is detected, redo detection from scratch and produce a corrected list.
                        - Self-check the final array before output: uniqueness, visibility, scope, correct control type, forbidden patterns absent, no iframe traversal.
                        """
                    
                    result = generate_model(
                        "You are an expert at analyzing HTML and identifying form inputs. Return only valid JSON with CSS selectors.",
                        model_object_type=CheckboxApplicationFieldResponse,
                        system_prompt=system_prompt,
                        model="gpt-5-mini"
                    ).fields
                    
                    if result:
                        results.append(result)
                        if spinner:
                            spinner.write(f"🔍 Run {i+1} found {len(result)} fields")
                        else:
                            print(f"🔍 Run {i+1} found {len(result)} fields")
                    else:
                        if spinner:
                            spinner.write(f"🔍 Run {i+1} returned empty result")
                        else:
                            print(f"🔍 Run {i+1} returned empty result")
                        
                except Exception as e:
                    if spinner:
                        spinner.write(f"❌ Error in run {i+1}: {e}")
                    else:
                        print(f"❌ Error in run {i+1}: {e}")
                    continue
            
            if not results:
                if spinner:
                    spinner.write(f"❌ All {best_of} runs failed, returning original fields")
                else:
                    print(f"❌ All {best_of} runs failed, returning original fields")
                return checkbox_fields
            
            # Find most common result by comparing field selectors
            from collections import Counter
            selector_counts = Counter()
            
            for result in results:
                for field in result:
                    if field.field_is_visible and field.field_in_form and field.field_type == FieldType.CHECKBOX:
                        selector_counts[field.field_selector] += 1
            
            # Get the most common selectors
            most_common_selectors = selector_counts.most_common()
            
            if spinner:
                spinner.write(f"🔍 Consensus results: {dict(selector_counts)}")
            else:
                print(f"🔍 Consensus results: {dict(selector_counts)}")
            
            # If we have consensus on most selectors, use the most common result
            if most_common_selectors and most_common_selectors[0][1] > 1:
                # Find the result with the most common selectors
                best_result = max(results, key=lambda r: sum(selector_counts.get(f.field_selector, 0) for f in r if f.field_is_visible and f.field_in_form and f.field_type == FieldType.CHECKBOX))
                
                best_fields = [field for field in best_result if field.field_is_visible and field.field_in_form and field.field_type == FieldType.CHECKBOX]
                
                if spinner:
                    spinner.write(f"🔍 Using consensus result with {len(best_fields)} fields")
                else:
                    print(f"🔍 Using consensus result with {len(best_fields)} fields")
                
                return best_fields
            else:
                if spinner:
                    spinner.write(f"🔍 No clear consensus, returning last result with {len(results[-1]) if results else 0} fields")
                else:
                    print(f"🔍 No clear consensus, returning last result with {len(results[-1]) if results else 0} fields")
                return results[-1] if results else checkbox_fields
        except Exception as e:
            print(f"❌ Error in verify_checkbox_fields: {e}")
            return []

    def verify_radio_fields(self, radio_fields: List[RadioApplicationField], frame=None, iframe_context=None, spinner:Yaspin=None, best_of: int = 3) -> List[RadioApplicationField]:
        """
        Verify the radio fields are valid using best_of parameter for consensus
        """
        try:
            if frame:
                page_content = frame.content()
            else:
                page_content = self.page.content()
            
            if best_of <= 1:
                # Single run - use original logic
                system_prompt = f"""
                    ROLE
                    You are a senior HTML/DOM form analyst. Verify a proposed list of RADIO fields against the actual page. If every item is valid, return the input list UNCHANGED. If any item fails, discard the list and RECOMPUTE the correct list from the HTML/screenshot. Output ONLY the final JSON array.

                    INPUTS
                    - html: {page_content}
                    - preferences: {self.preferences}
                    - proposed_radio_fields: {json.dumps([self._serialize_field_for_json(field) for field in radio_fields])}
                    - NOTE: {"The proposed radio fields list is empty - you MUST find all RADIO fields from scratch" if not radio_fields else "The proposed radio fields exist and should be validated"}

                    WHAT QUALIFIES AS A RADIO BUTTON
                    - Native <input type="radio"> elements
                    - Custom radio implementations with role="radio" and proper ARIA attributes
                    - Radio button groups that allow single selection from multiple options

                    ACTIVE SCOPE & VISIBILITY
                    - If a visible modal/dialog is open (role="dialog" or [aria-modal="true"]), TREAT IT AS THE ONLY ACTIVE SCOPE; ignore background fields.
                    - Include ONLY elements that are visible & interactive (not display:none, visibility:hidden, opacity:0, off-screen/zero-size, [hidden], [aria-hidden="true"], [disabled], or aria-disabled="true"]).
                    - Include ONLY elements that belong to the job-application form (a relevant <form> ancestor).
                    - Do NOT cross iframe boundaries in selectors. If a field is inside an iframe, write the selector as if already inside that iframe DOM (no iframe prefix).

                    FORBIDDEN (NEVER VALID)
                    - Selectors that include iframe nodes (e.g., iframe#... ...)
                    - Labels/spans instead of the control: label[for=...], span.field-label
                    - Checkboxes: input[type="checkbox"]
                    - Text inputs: input[type="text"], input[type="email"], etc.
                    - Overly generic roots: html body input, html body div
                    - Positional/transient-state selectors: :nth-*, [aria-expanded=...], inline style predicates, off-screen portals
                    - Hidden/disabled elements
                    - Non-radio controls: buttons, links, selects, textareas

                    VALIDATION CHECKLIST (APPLY TO EACH proposed item)
                    1) Selector resolves to EXACTLY ONE element in the active scope.
                    2) Element is a valid radio control:
                    - Native: tagName == 'INPUT' and type == 'radio'
                    - Custom: has role="radio" or equivalent radio semantics
                    3) Element is visible & interactive now; not hidden/disabled/off-screen/future-step.
                    4) Element is inside the job-application form (or active modal form).
                    5) No iframe traversal in the selector.
                    6) field_type == "radio"; field_is_visible == true; field_in_form == true.
                    7) field_selector is minimal & stable (prefer id/name/data-*; avoid positional/state-based hacks).
                    8) field_value is boolean; field_group_name is string; field_options is array.

                    DECISION RULE
                    - If ALL proposed items pass EVERY validation and no qualifying radio buttons are missing, RETURN THE proposed_fields_json UNCHANGED (identical order and text).
                    - Otherwise, DISCARD proposed_fields_json and RECOMPUTE:
                    a) Re-detect all qualifying radio buttons from html/screenshot per the rules above.
                    b) Build minimal, stable selectors (native <input type="radio"> or custom radio with role="radio").
                    c) Order by top-to-bottom DOM order within the active scope.
                    d) Exclude anything forbidden.

                    OUTPUT
                    - Output ONLY the final JSON array of ApplicationField objects. NO extra text.
                    - If nothing qualifies, output [].

                    REASONING PROTOCOL (DO NOT REVEAL)
                    - Silently validate proposed_fields_json against the checklist.
                    - If any failure or missing field is detected, redo detection from scratch and produce a corrected list.
                    - Self-check the final array before output: uniqueness, visibility, scope, correct control type, forbidden patterns absent, no iframe traversal.
                    """
                
                all_fields = generate_model(
                    "You are an expert at analyzing HTML and identifying form inputs. Return only valid JSON with CSS selectors.",
                    model_object_type=RadioApplicationFieldResponse,
                    system_prompt=system_prompt,
                    model="gpt-5-mini"
                ).fields
                
                all_fields = [field for field in all_fields if field.field_is_visible and field.field_in_form]
                if debug_mode:
                    spinner.write(f"✅ verified {len(all_fields)} fields")
                
                # Filter out fields that are not visible
                context_str = "iframe" if frame else "page"
                
                # Parse the JSON response
                radio_fields: List[RadioApplicationField] = [field for field in all_fields if field.field_type == FieldType.RADIO]
                
                context_str = "iframe" if frame else "page"
                if debug_mode:
                    spinner.write(f"✅ gpt-5-mini found {len(radio_fields)} radio fields in {context_str}")
                
                return radio_fields
            
            # Multiple runs for consensus
            results = []
            for i in range(best_of):
                try:
                    if spinner:
                        spinner.write(f"🔍 Run {i+1}/{best_of} for radio fields verification...")
                    else:
                        print(f"🔍 Run {i+1}/{best_of} for radio fields verification...")
                    
                    system_prompt = f"""
                        ROLE
                        You are a senior HTML/DOM form analyst. Verify a proposed list of RADIO fields against the actual page. If every item is valid, return the input list UNCHANGED. If any item fails, discard the list and RECOMPUTE the correct list from the HTML/screenshot. Output ONLY the final JSON array.

                        INPUTS
                        - html: {page_content}
                        - preferences: {self.preferences}
                        - proposed_radio_fields: {json.dumps([self._serialize_field_for_json(field) for field in radio_fields])}

                        TARGET SCHEMA (ApplicationField)
                        Each object MUST have:
                        - "field_selector": minimal, stable CSS selector for the radio control
                        - "field_name": name attribute or best-effort visible label
                        - "field_type": "radio"
                        - "field_is_visible": true only if visible & interactive now
                        - "field_in_form": true only if inside the application form (or active modal form)
                        - "field_group_name": name attribute value for radio group identification
                        - "field_options": array of visible options if available, else []

                        WHAT QUALIFIES AS A RADIO BUTTON
                        - Native <input type="radio"> elements
                        - Custom radio implementations with role="radio" and proper ARIA attributes
                        - Radio button groups that allow single selection from multiple options

                        ACTIVE SCOPE & VISIBILITY
                        - If a visible modal/dialog is open (role="dialog" or [aria-modal="true"]), TREAT IT AS THE ONLY ACTIVE SCOPE; ignore background fields.
                        - Include ONLY elements that are visible & interactive (not display:none, visibility:hidden, opacity:0, off-screen/zero-size, [hidden], [aria-hidden="true"], [disabled], or aria-disabled="true"]).
                        - Include ONLY elements that belong to the job-application form (a relevant <form> ancestor).
                        - Do NOT cross iframe boundaries in selectors. If a field is inside an iframe, write the selector as if already inside that iframe DOM (no iframe prefix).

                        FORBIDDEN (NEVER VALID)
                        - Selectors that include iframe nodes (e.g., iframe#... ...)
                        - Labels/spans instead of the control: label[for=...], span.field-label
                        - Checkboxes: input[type="checkbox"]
                        - Text inputs: input[type="text"], input[type="email"], etc.
                        - Overly generic roots: html body input, html body div
                        - Positional/transient-state selectors: :nth-*, [aria-expanded=...], inline style predicates, off-screen portals
                        - Hidden/disabled elements
                        - Non-radio controls: buttons, links, selects, textareas

                        VALIDATION CHECKLIST (APPLY TO EACH proposed item)
                        1) Selector resolves to EXACTLY ONE element in the active scope.
                        2) Element is a valid radio control:
                        - Native: tagName == 'INPUT' and type == 'radio'
                        - Custom: has role="radio" or equivalent radio semantics
                        3) Element is visible & interactive now; not hidden/disabled/off-screen/future-step.
                        4) Element is inside the job-application form (or active modal form).
                        5) No iframe traversal in the selector.
                        6) field_type == "radio"; field_is_visible == true; field_in_form == true.
                        7) field_selector is minimal & stable (prefer id/name/data-*; avoid positional/state-based hacks).
                        8) field_value is boolean; field_group_name is string; field_options is array.

                        DECISION RULE
                        - If ALL proposed items pass EVERY validation and no qualifying radio buttons are missing, RETURN THE proposed_fields_json UNCHANGED (identical order and text).
                        - Otherwise, DISCARD proposed_fields_json and RECOMPUTE:
                        a) Re-detect all qualifying radio buttons from html/screenshot per the rules above.
                        b) Build minimal, stable selectors (native <input type="radio"> or custom radio with role="radio").
                        c) Order by top-to-bottom DOM order within the active scope.
                        d) Exclude anything forbidden.

                        OUTPUT
                        - Output ONLY the final JSON array of ApplicationField objects. NO extra text.
                        - If nothing qualifies, output [].

                        REASONING PROTOCOL (DO NOT REVEAL)
                        - Silently validate proposed_fields_json against the checklist.
                        - If any failure or missing field is detected, redo detection from scratch and produce a corrected list.
                        - Self-check the final array before output: uniqueness, visibility, scope, correct control type, forbidden patterns absent, no iframe traversal.
                        """
                    
                    result = generate_model(
                        "You are an expert at analyzing HTML and identifying form inputs. Return only valid JSON with CSS selectors.",
                        model_object_type=RadioApplicationFieldResponse,
                        system_prompt=system_prompt,
                        model="gpt-5-mini"
                    ).fields
                    
                    if result:
                        results.append(result)
                        if spinner:
                            spinner.write(f"🔍 Run {i+1} found {len(result)} fields")
                        else:
                            print(f"🔍 Run {i+1} found {len(result)} fields")
                    else:
                        if spinner:
                            spinner.write(f"🔍 Run {i+1} returned empty result")
                        else:
                            print(f"🔍 Run {i+1} returned empty result")
                        
                except Exception as e:
                    if spinner:
                        spinner.write(f"❌ Error in run {i+1}: {e}")
                    else:
                        print(f"❌ Error in run {i+1}: {e}")
                    continue
            
            if not results:
                if spinner:
                    spinner.write(f"❌ All {best_of} runs failed, returning original fields")
                else:
                    print(f"❌ All {best_of} runs failed, returning original fields")
                return radio_fields
            
            # Find most common result by comparing field selectors
            from collections import Counter
            selector_counts = Counter()
            
            for result in results:
                for field in result:
                    if field.field_is_visible and field.field_in_form and field.field_type == FieldType.RADIO:
                        selector_counts[field.field_selector] += 1
            
            # Get the most common selectors
            most_common_selectors = selector_counts.most_common()
            
            if spinner:
                spinner.write(f"🔍 Consensus results: {dict(selector_counts)}")
            else:
                print(f"🔍 Consensus results: {dict(selector_counts)}")
            
            # If we have consensus on most selectors, use the most common result
            if most_common_selectors and most_common_selectors[0][1] > 1:
                # Find the result with the most common selectors
                best_result = max(results, key=lambda r: sum(selector_counts.get(f.field_selector, 0) for f in r if f.field_is_visible and f.field_in_form and f.field_type == FieldType.RADIO))
                
                best_fields = [field for field in best_result if field.field_is_visible and field.field_in_form and field.field_type == FieldType.RADIO]
                
                if spinner:
                    spinner.write(f"🔍 Using consensus result with {len(best_fields)} fields")
                else:
                    print(f"🔍 Using consensus result with {len(best_fields)} fields")
                
                return best_fields
            else:
                if spinner:
                    spinner.write(f"🔍 No clear consensus, returning last result with {len(results[-1]) if results else 0} fields")
                else:
                    print(f"🔍 No clear consensus, returning last result with {len(results[-1]) if results else 0} fields")
                return results[-1] if results else radio_fields
        except Exception as e:
            print(f"❌ Error in verify_radio_fields: {e}")
            return []

    def verify_text_input_fields(self, text_fields: List[TextApplicationField], frame=None, iframe_context=None, spinner:Yaspin=None, best_of: int = 3) -> List[TextApplicationField]:
        """
        Verify the text input fields are valid using best_of parameter for consensus
        """
        try:
            if frame:
                page_content = frame.content()
            else:
                page_content = self.page.content()
            
            if best_of <= 1:
                # Single run - use original logic
                system_prompt = f"""
                    ROLE
                    You are a senior HTML/DOM form analyst. Verify a proposed list of TEXT INPUT fields against the actual page. If every item is valid, return the input list UNCHANGED. If any item fails, discard the list and RECOMPUTE the correct list from the HTML/screenshot. Output ONLY the final JSON array.

                    INPUTS
                    - html: {page_content}
                    - preferences: {self.preferences}
                    - proposed_text_fields: {json.dumps([self._serialize_field_for_json(field) for field in text_fields])}
                    - NOTE: {"The proposed text input fields list is empty - you MUST find all TEXT INPUT fields from scratch" if not text_fields else "The proposed text input fields exist and should be validated"}


                    TARGET SCHEMA (ApplicationField)
                    Each object MUST have:
                    - "field_selector": minimal, stable CSS selector for the text input control
                    - "field_name": name attribute or best-effort visible label
                    - "field_type": "text"
                    - "field_is_visible": true only if visible & interactive now
                    - "field_in_form": true only if inside the application form (or active modal form)
                    - "field_value": "" (empty string for text inputs)
                    - "field_max_length": maximum character limit if available, else null
                    - "field_pattern": input pattern if available, else null
                    - "field_placeholder": placeholder text if available, else null

                    WHAT QUALIFIES AS A TEXT INPUT
                    - Native <input> types: text, email, tel, url, search, number, password, date, datetime-local, time
                    - <textarea> elements
                    - Contenteditable widgets with stable input/textarea-equivalent role (avoid unless clearly used for text entry)

                    ACTIVE SCOPE & VISIBILITY
                    - If a visible modal/dialog is open (role="dialog" or [aria-modal="true"]), TREAT IT AS THE ONLY ACTIVE SCOPE; ignore background fields.
                    - Include ONLY elements that are visible & interactive (not display:none, visibility:hidden, opacity:0, off-screen/zero-size, [hidden], [aria-hidden="true"], [disabled], or aria-disabled="true"]).
                    - Include ONLY elements that belong to the job-application form (a relevant <form> ancestor).
                    - Do NOT cross iframe boundaries in selectors. If a field is inside an iframe, write the selector as if already inside that iframe DOM (no iframe prefix).

                    FORBIDDEN (NEVER VALID)
                    - Selectors that include iframe nodes (e.g., iframe#... ...)
                    - Labels/spans instead of the control: label[for=...], span.field-label
                    - Checkboxes: input[type="checkbox"]
                    - Radio buttons: input[type="radio"]
                    - File inputs: input[type="file"]
                    - Select dropdowns: <select> elements
                    - Overly generic roots: html body input, html body textarea
                    - Positional/transient-state selectors: :nth-*, [aria-expanded=...], inline style predicates, off-screen portals
                    - Hidden/disabled elements
                    - Non-text controls: buttons, links, selects

                    VALIDATION CHECKLIST (APPLY TO EACH proposed item)
                    1) Selector resolves to EXACTLY ONE element in the active scope.
                    2) Element is a valid text input control:
                    - Native: tagName == 'INPUT' with appropriate type OR tagName == 'TEXTAREA'
                    - Custom: has role equivalent to text input or textarea
                    3) Element is visible & interactive now; not hidden/disabled/off-screen/future-step.
                    4) Element is inside the job-application form (or active modal form).
                    5) No iframe traversal in the selector.
                    6) field_type == "text"; field_is_visible == true; field_in_form == true.
                    7) field_selector is minimal & stable (prefer id/name/data-*; avoid positional/state-based hacks).
                    8) field_value is string; field_max_length, field_pattern, field_placeholder are appropriate types.

                    DECISION RULE
                    - If ALL proposed items pass EVERY validation and no qualifying text inputs are missing, RETURN THE proposed_fields_json UNCHANGED (identical order and text).
                    - Otherwise, DISCARD proposed_fields_json and RECOMPUTE:
                    a) Re-detect all qualifying text inputs from html/screenshot per the rules above.
                    b) Build minimal, stable selectors (native <input> or <textarea>).
                    c) Order by top-to-bottom DOM order within the active scope.
                    d) Exclude anything forbidden.

                    OUTPUT
                    - Output ONLY the final JSON array of ApplicationField objects. NO extra text.
                    - If nothing qualifies, output [].

                    REASONING PROTOCOL (DO NOT REVEAL)
                    - Silently validate proposed_fields_json against the checklist.
                    - If any failure or missing field is detected, redo detection from scratch and produce a corrected list.
                    - Self-check the final array before output: uniqueness, visibility, scope, correct control type, forbidden patterns absent, no iframe traversal.
                    """
                
                all_fields = generate_model(
                    "You are an expert at analyzing HTML and identifying form inputs. Return only valid JSON with CSS selectors.",
                    model_object_type=TextApplicationFieldResponse,
                    system_prompt=system_prompt,
                    model="gpt-5-mini"
                ).fields
                
                all_fields = [field for field in all_fields if field.field_is_visible and field.field_in_form]
                if debug_mode:
                    spinner.write(f"✅ verified {len(all_fields)} fields")
                
                # Filter out fields that are not visible
                context_str = "iframe" if frame else "page"
                
                # Parse the JSON response
                text_fields: List[TextApplicationField] = [field for field in all_fields if field.field_type == FieldType.TEXT]
                
                context_str = "iframe" if frame else "page"
                if debug_mode:
                    spinner.write(f"✅ gpt-5-mini found {len(text_fields)} text input fields in {context_str}")
                
                return text_fields
            
            # Multiple runs for consensus
            results = []
            for i in range(best_of):
                try:
                    if spinner:
                        spinner.write(f"🔍 Run {i+1}/{best_of} for text input fields verification...")
                    else:
                        print(f"🔍 Run {i+1}/{best_of} for text input fields verification...")
                    
                    system_prompt = f"""
                        ROLE
                        You are a senior HTML/DOM form analyst. Verify a proposed list of TEXT INPUT fields against the actual page. If every item is valid, return the input list UNCHANGED. If any item fails, discard the list and RECOMPUTE the correct list from the HTML/screenshot. Output ONLY the final JSON array.

                        INPUTS
                        - html: {page_content}
                        - preferences: {self.preferences}
                        - proposed_text_fields: {json.dumps([self._serialize_field_for_json(field) for field in text_fields])}

                        TARGET SCHEMA (ApplicationField)
                        Each object MUST have:
                        - "field_selector": minimal, stable CSS selector for the text input control
                        - "field_name": name attribute or best-effort visible label
                        - "field_type": "text"
                        - "field_is_visible": true only if visible & interactive now
                        - "field_in_form": true only if inside the application form (or active modal form)
                        - "field_value": "" (empty string for text inputs)
                        - "field_max_length": maximum character limit if available, else null
                        - "field_pattern": input pattern if available, else null
                        - "field_placeholder": placeholder text if available, else null

                        WHAT QUALIFIES AS A TEXT INPUT
                        - Native <input> types: text, email, tel, url, search, number, password, date, datetime-local, time
                        - <textarea> elements
                        - Contenteditable widgets with stable input/textarea-equivalent role (avoid unless clearly used for text entry)

                        ACTIVE SCOPE & VISIBILITY
                        - If a visible modal/dialog is open (role="dialog" or [aria-modal="true"]), TREAT IT AS THE ONLY ACTIVE SCOPE; ignore background fields.
                        - Include ONLY elements that are visible & interactive (not display:none, visibility:hidden, opacity:0, off-screen/zero-size, [hidden], [aria-hidden="true"], [disabled], or aria-disabled="true"]).
                        - Include ONLY elements that belong to the job-application form (a relevant <form> ancestor).
                        - Do NOT cross iframe boundaries in selectors. If a field is inside an iframe, write the selector as if already inside that iframe DOM (no iframe prefix).

                        FORBIDDEN (NEVER VALID)
                        - Selectors that include iframe nodes (e.g., iframe#... ...)
                        - Labels/spans instead of the control: label[for=...], span.field-label
                        - Checkboxes: input[type="checkbox"]
                        - Radio buttons: input[type="radio"]
                        - File inputs: input[type="file"]
                        - Select dropdowns: <select> elements
                        - Overly generic roots: html body input, html body textarea
                        - Positional/transient-state selectors: :nth-*, [aria-expanded=...], inline style predicates, off-screen portals
                        - Hidden/disabled elements
                        - Non-text controls: buttons, links, selects

                        VALIDATION CHECKLIST (APPLY TO EACH proposed item)
                        1) Selector resolves to EXACTLY ONE element in the active scope.
                        2) Element is a valid text input control:
                        - Native: tagName == 'INPUT' with appropriate type OR tagName == 'TEXTAREA'
                        - Custom: has role equivalent to text input or textarea
                        3) Element is visible & interactive now; not hidden/disabled/off-screen/future-step.
                        4) Element is inside the job-application form (or active modal form).
                        5) No iframe traversal in the selector.
                        6) field_type == "text"; field_is_visible == true; field_in_form == true.
                        7) field_selector is minimal & stable (prefer id/name/data-*; avoid positional/state-based hacks).
                        8) field_value is string; field_max_length, field_pattern, field_placeholder are appropriate types.

                        DECISION RULE
                        - If ALL proposed items pass EVERY validation and no qualifying text inputs are missing, RETURN THE proposed_fields_json UNCHANGED (identical order and text).
                        - Otherwise, DISCARD proposed_fields_json and RECOMPUTE:
                        a) Re-detect all qualifying text inputs from html/screenshot per the rules above.
                        b) Build minimal, stable selectors (native <input> or <textarea>).
                        c) Populate field_name (best-effort), field_type="text", field_is_visible=true, field_in_form=true, field_value="", and other properties when available.
                        d) Order by top-to-bottom DOM order within the active scope.
                        e) Exclude anything forbidden.

                        OUTPUT
                        - Output ONLY the final JSON array of ApplicationField objects. NO extra text.
                        - If nothing qualifies, output [].

                        REASONING PROTOCOL (DO NOT REVEAL)
                        - Silently validate proposed_fields_json against the checklist.
                        - If any failure or missing field is detected, redo detection from scratch and produce a corrected list.
                        - Self-check the final array before output: uniqueness, visibility, scope, correct control type, forbidden patterns absent, no iframe traversal.
                        """
                    
                    result = generate_model(
                        "You are an expert at analyzing HTML and identifying form inputs. Return only valid JSON with CSS selectors.",
                        model_object_type=TextApplicationFieldResponse,
                        system_prompt=system_prompt,
                        model="gpt-5-mini"
                    ).fields
                    
                    if result:
                        results.append(result)
                        if spinner:
                            spinner.write(f"🔍 Run {i+1} found {len(result)} fields")
                        else:
                            print(f"🔍 Run {i+1} found {len(result)} fields")
                    else:
                        if spinner:
                            spinner.write(f"🔍 Run {i+1} returned empty result")
                        else:
                            print(f"🔍 Run {i+1} returned empty result")
                        
                except Exception as e:
                    if spinner:
                        spinner.write(f"❌ Error in run {i+1}: {e}")
                    else:
                        print(f"❌ Error in run {i+1}: {e}")
                    continue
            
            if not results:
                if spinner:
                    spinner.write(f"❌ All {best_of} runs failed, returning original fields")
                else:
                    print(f"❌ All {best_of} runs failed, returning original fields")
                return text_fields
            
            # Find most common result by comparing field selectors
            from collections import Counter
            selector_counts = Counter()
            
            for result in results:
                for field in result:
                    if field.field_is_visible and field.field_in_form and field.field_type == FieldType.TEXT:
                        selector_counts[field.field_selector] += 1
            
            # Get the most common selectors
            most_common_selectors = selector_counts.most_common()
            
            if spinner:
                spinner.write(f"🔍 Consensus results: {dict(selector_counts)}")
            else:
                print(f"🔍 Consensus results: {dict(selector_counts)}")
            
            # If we have consensus on most selectors, use the most common result
            if most_common_selectors and most_common_selectors[0][1] > 1:
                # Find the result with the most common selectors
                best_result = max(results, key=lambda r: sum(selector_counts.get(f.field_selector, 0) for f in r if f.field_is_visible and f.field_in_form and f.field_type == FieldType.TEXT))
                
                best_fields = [field for field in best_result if field.field_is_visible and field.field_in_form and field.field_type == FieldType.TEXT]
                
                if spinner:
                    spinner.write(f"🔍 Using consensus result with {len(best_fields)} fields")
                else:
                    print(f"🔍 Using consensus result with {len(best_fields)} fields")
                
                return best_fields
            else:
                if spinner:
                    spinner.write(f"🔍 No clear consensus, returning last result with {len(results[-1]) if results else 0} fields")
                else:
                    print(f"🔍 No clear consensus, returning last result with {len(results[-1]) if results else 0} fields")
                return results[-1] if results else text_fields
        except Exception as e:
            print(f"❌ Error in verify_text_input_fields: {e}")
            return []

    def _get_file_path_for_field(self, field_name_lower: str) -> str:
        """Get the appropriate file path based on field name"""
        if 'resume' in field_name_lower or 'cv' in field_name_lower:
            file_path = self.preferences.get('resume_path')
            print(f"📄 Resume field detected: {field_name_lower}")
        elif 'cover' in field_name_lower or 'letter' in field_name_lower:
            file_path = self.preferences.get('cover_letter_path')
            print(f"📝 Cover letter field detected: {field_name_lower}")
        elif 'photo' in field_name_lower or 'image' in field_name_lower:
            file_path = self.preferences.get('photo_path')
            print(f"📷 Photo field detected: {field_name_lower}")
        else:
            # Try to find any file path in preferences
            file_path = None
            for key, value in self.preferences.items():
                if 'path' in key.lower() and value and isinstance(value, str):
                    file_path = value
                    print(f"📁 Using file path from preference '{key}': {field_name_lower}")
                    break
        
        return file_path

    def _take_smart_screenshot(self, frame=None, iframe_context=None, spinner: Yaspin=None):
        """
        Take a screenshot using Chrome DevTools (CDP). No page.screenshot() and no new tabs.
        - Default: full-page capture of the main document.
        - If an iframe is provided, capture the iframe's VISIBLE region (clip) on the main page.

        Returns:
            (screenshot_bytes: bytes, context_str: str)
        """
        import base64
        import math

        def _cdp_session(p):
            return p.context.new_cdp_session(p)

        def _cdp_fullpage_png(p) -> bytes:
            cdp = _cdp_session(p)
            metrics = cdp.send("Page.getLayoutMetrics")
            cs = metrics["contentSize"]  # dict: x, y, width, height
            w = max(1, int(math.ceil(cs.get("width", 1))))
            h = max(1, int(math.ceil(cs.get("height", 1))))

            # Keep within safe texture limits
            max_dim = 16384
            scale = 1.0
            if max(w, h) > max_dim:
                scale = max_dim / float(max(w, h))

            data_b64 = cdp.send("Page.captureScreenshot", {
                "format": "png",
                "fromSurface": True,
                "captureBeyondViewport": True,
                "clip": {"x": 0, "y": 0, "width": w, "height": h, "scale": scale},
            })["data"]
            return base64.b64decode(data_b64)

        def _cdp_clip_png(p, x, y, width, height, scale=1.0) -> bytes:
            cdp = _cdp_session(p)
            data_b64 = cdp.send("Page.captureScreenshot", {
                "format": "png",
                "fromSurface": True,
                "captureBeyondViewport": True,
                "clip": {"x": float(x), "y": float(y), "width": float(width), "height": float(height), "scale": float(scale)},
            })["data"]
            return base64.b64decode(data_b64)

        def _try_iframe_clip() -> tuple | None:
            """
            If we can identify an iframe element, return (x, y, w, h) in document coords.
            Returns None if not found or not visible.
            """
            # 1) Prefer explicit iframe element in iframe_context
            if iframe_context and isinstance(iframe_context, dict) and "iframe" in iframe_context:
                el = iframe_context["iframe"]
            # 2) Else, if a frame handle is provided, map it to its <iframe> element
            elif frame:
                el = None
                try:
                    for cand in self.page.query_selector_all("iframe"):
                        try:
                            if cand.content_frame() == frame:
                                el = cand
                                break
                        except Exception:
                            continue
                except Exception:
                    el = None
                if el is None:
                    return None
            else:
                return None

            try:
                # Ensure it's rendered and get a bounding box
                el.scroll_into_view_if_needed()
                bbox = el.bounding_box()
                if not bbox:
                    return None

                # Translate viewport coords -> document coords
                scroll = self.page.evaluate("""() => ({x: window.scrollX || 0, y: window.scrollY || 0})""")
                x_doc = bbox["x"] + scroll["x"]
                y_doc = bbox["y"] + scroll["y"]
                return (x_doc, y_doc, bbox["width"], bbox["height"])
            except Exception:
                return None

        try:
            # If we can identify an iframe element, capture its visible region only.
            clip_rect = _try_iframe_clip()
            if clip_rect:
                if spinner and 'debug_mode' in globals() and debug_mode:
                    spinner.write("📸 Capturing iframe VISIBLE region via CDP (no new tab).")
                x, y, w, h = clip_rect
                screenshot = _cdp_clip_png(self.page, x, y, w, h)
                context_str = "iframe_region"
            else:
                # Default: full page capture
                if spinner and 'debug_mode' in globals() and debug_mode:
                    spinner.write("📸 Capturing MAIN PAGE full document via CDP.")
                screenshot = _cdp_fullpage_png(self.page)
                context_str = "main_page_full"

            # Persist (optional)
            filename = f"screenshot_{context_str}.png"
            with open(filename, "wb") as f:
                f.write(screenshot)

            return screenshot, context_str

        except Exception as e:
            dprint(f"❌ CDP capture error: {e}")
            # Last-resort: viewport-only CDP capture
            try:
                cdp = self.page.context.new_cdp_session(self.page)
                data_b64 = cdp.send("Page.captureScreenshot", {"format": "png"})["data"]
                screenshot = base64.b64decode(data_b64)
                with open("screenshot_fallback.png", "wb") as f:
                    f.write(screenshot)
                dprint("💾 Fallback viewport screenshot saved: screenshot_fallback.png")
                return screenshot, "viewport_fallback"
            except Exception as fallback_error:
                dprint(f"❌ Critical error: Could not take any screenshot via CDP: {fallback_error}")
                return None, "error"
    
    def find_all_form_inputs(self, frame=None, iframe_context=None, spinner:Yaspin=None) -> Tuple[List[TextApplicationField], List[SelectApplicationField], List[RadioApplicationField], List[CheckboxApplicationField], List[UploadApplicationField]]:
        """
        Find all types of form inputs on the current page or in iframe using unified field detection
        
        Args:
            frame: Optional iframe frame context. If None, uses main page.
            
        Returns:
            Tuple with categorized fields
        """
        try:
            # Use unified field detection functions
            text_input_fields = self.find_all_text_input_fields(frame, iframe_context, spinner)
            spinner.write(f"Found {len(text_input_fields)} text input fields")
            selectors = self.find_all_select_fields(frame, iframe_context, spinner)
            spinner.write(f"Found {len(selectors)} select fields")
            radio_groups = self.find_all_radio_fields(frame, iframe_context, spinner)
            spinner.write(f"Found {len(radio_groups)} radio groups")
            checkboxes = self.find_all_checkbox_fields(frame, iframe_context, spinner)
            spinner.write(f"Found {len(checkboxes)} checkboxes")
            
            # For upload buttons, we need to handle the single return value
            upload_button = self.find_upload_file_button(frame, iframe_context, spinner)
            upload_fields = [upload_button] if upload_button is not None else []
            
            total_fields = (len(text_input_fields) + len(selectors) + len(radio_groups) + 
                            len(checkboxes) + len(upload_fields))
            
            context_str = "iframe" if frame else "page"
            dprint(f"✅ Unified field detection found {total_fields} total form fields in {context_str}")
            dprint(f"  - {len(text_input_fields)} text input fields")
            dprint(f"  - {len(selectors)} select dropdowns")
            dprint(f"  - {len(radio_groups)} radio button groups")
            dprint(f"  - {len(checkboxes)} checkboxes")
            dprint(f"  - {len(upload_fields)} upload buttons")
            
            # Categorize fields by type
            text_fields = []
            select_fields = []
            radio_fields = []
            checkbox_fields = []
            upload_fields_list = []
            
            # Process text fields
            for field in text_input_fields:
                if isinstance(field, TextApplicationField):
                    text_fields.append(field)
                else:
                    # Convert generic field to TextApplicationField
                    text_field = TextApplicationField(
                        field_order=field.field_order,
                        field_name=field.field_name,
                        field_selector=field.field_selector,
                        field_is_visible=field.field_is_visible,
                        field_in_form=field.field_in_form,
                        field_required=field.field_required,
                        field_placeholder=field.field_placeholder,
                        field_value=field.field_value
                    )
                    text_fields.append(text_field)
            
            # Process select fields
            for field in selectors:
                if isinstance(field, SelectApplicationField):
                    select_fields.append(field)
                else:
                    # Convert generic field to SelectApplicationField
                    select_field = SelectApplicationField(
                        field_order=field.field_order,
                        field_name=field.field_name,
                        field_selector=field.field_selector,
                        field_is_visible=field.field_is_visible,
                        field_in_form=field.field_in_form,
                        field_required=field.field_required,
                        field_placeholder=field.field_placeholder,
                        field_value=field.field_value,
                        field_options=field.field_options
                    )
                    select_fields.append(select_field)
            
            # Process radio fields
            for field in radio_groups:
                if isinstance(field, RadioApplicationField):
                    radio_fields.append(field)
                else:
                    # Convert generic field to RadioApplicationField
                    radio_field = RadioApplicationField(
                        field_order=field.field_order,
                        field_name=field.field_name,
                        field_selector=field.field_selector,
                        field_is_visible=field.field_is_visible,
                        field_in_form=field.field_in_form,
                        field_required=field.field_required,
                        field_placeholder=field.field_placeholder,
                        field_value=field.field_value,
                        field_options=field.field_options
                    )
                    radio_fields.append(radio_field)
            
            # Process checkbox fields
            for field in checkboxes:
                if isinstance(field, CheckboxApplicationField):
                    checkbox_fields.append(field)
                else:
                    # Convert generic field to CheckboxApplicationField
                    checkbox_field = CheckboxApplicationField(
                        field_order=field.field_order,
                        field_name=field.field_name,
                        field_selector=field.field_selector,
                        field_is_visible=field.field_is_visible,
                        field_in_form=field.field_in_form,
                        field_required=field.field_required,
                        field_placeholder=field.field_placeholder,
                        field_value=field.field_value
                    )
                    checkbox_fields.append(checkbox_field)
            
            # Process upload fields
            for field in upload_fields:
                if isinstance(field, UploadApplicationField):
                    upload_fields_list.append(field)
                else:
                    # Convert generic field to UploadApplicationField
                    upload_field = UploadApplicationField(
                        field_order=field.field_order,
                        field_name=field.field_name,
                        field_selector=field.field_selector,
                        field_is_visible=field.field_is_visible,
                        field_in_form=field.field_in_form,
                        field_required=field.field_required,
                        field_placeholder=field.field_placeholder,
                        field_value=field.field_value
                    )
                    upload_fields_list.append(upload_field)
            
            return text_fields, select_fields, radio_fields, checkbox_fields, upload_fields_list
            
        except Exception as e:
            context_str = "iframe" if frame else "page"
            dprint(f"❌ Error finding form fields with unified detection in {context_str}: {e}")
            return [], [], [], [], []

    def _find_alternative_select_value(self, possible_values: List[str], select_field: SelectApplicationField, preferences: Dict[str, Any]) -> Optional[str]:
        """
        Find an alternative value from the available options when the original value is not found
        
        Args:
            possible_values: List of available values in the select field
            select_field: The select field that needs an alternative value
            preferences: User preferences for filling forms
            
        Returns:
            Optional[str]: Alternative value if found, None otherwise
        """
        try:
            field_name = select_field.field_name or ""
            page_content = self.page.content()
            print(f"🤖 Using gpt-5-mini to find alternative value for field: {field_name}")
            print(f"   Available values: {possible_values}")
            
            system_prompt = f"""
            You are helping to fill out a job application form. A select field has an invalid value that needs to be replaced with one of the available options.
            You are given the html of the page: {page_content}
            
            FIELD INFORMATION:
            - Field name: {field_name}
            - Possible options: {possible_values}
            
            USER PREFERENCES:
            {preferences}
            
            INSTRUCTIONS:
            1. Analyze the field name and understand what type of information it's asking for
            2. Consider the user's preferences to find the most suitable option
            3. Choose the best alternative from the available options
            4. If no good match exists, choose the most reasonable default option
            5. Avoid placeholder values like "Select", "Choose", "--", etc.
            
            Return only the selected value as a string, nothing else.
            
            Examples:
            - If field is "location" and user prefers "London" but only "Remote" is available, return "Remote"
            - If field is "experience_level" and user prefers "Senior" but only "Mid-level" is available, return "Mid-level"
            - If field is "employment_type" and user prefers "Full-time" but only "Permanent" is available, return "Permanent"
            """
            
            gemini_model = "gpt-5-mini"
            response = generate_text("You are an expert at analyzing form fields and selecting appropriate values. Return only the selected value as a string.", system_prompt=system_prompt, model=gemini_model)
            
            selected_value = response.strip()
            
            if not possible_values:
                return selected_value
            
            # Verify the selected value is actually in the available options
            if selected_value in possible_values:
                dprint(f"✅ {gemini_model} selected: {selected_value}")
                return selected_value
            else:
                # Try case-insensitive match
                for value in possible_values:
                    if value.lower() == selected_value.lower():
                        dprint(f"✅ {gemini_model} selected (case-insensitive): {value}")
                        return value
                
                # If still no match, use the first valid option
                for value in possible_values:
                    if value and value.strip() and value.lower() not in ['select', 'choose', 'please select', '--', '']:
                        dprint(f"⚠️ {gemini_model} selection '{selected_value}' not found, using fallback: {value}")
                        return value
            
            print(f"❌ No suitable alternative value found for {field_name}")
            return None
            
        except Exception as e:
            dprint(f"❌ Error in {gemini_model} alternative value selection: {e}")
            # Fallback to first valid option
            for value in possible_values:
                if value and value.strip() and value.lower() not in ['select', 'choose', 'please select', '--', '']:
                    dprint(f"🔄 Using fallback value: {value}")
                    return value
            return None

    def click_and_type_in_field(self, field: TextApplicationField, text: str, frame=None) -> bool:
        """Click on a field and type text using Playwright in page or iframe context"""
        try:
            # Use id= selector engine for IDs with special characters like []
            if field.field_selector.startswith('#'):
                if frame:
                    element = frame.locator(f'id={field.field_selector[1:]}').first
                else:
                    element = self.page.locator(f'id={field.field_selector[1:]}').first
            else:
                if frame:
                    element = frame.locator(field.field_selector)
                else:
                    element = self.page.locator(field.field_selector)
            
            if not element:
                print(f"⚠️ Element not found with selector: {field.field_selector}")
                return False
            
            # Check if element is interactive before attempting to interact
            if not self._is_element_interactive(element, frame):
                context_str = "iframe" if frame else "page"
                print(f"⏭️ Skipping non-interactive text field in {context_str}: {field.field_name}")
                return True  # Return True to continue with other fields
            
            # Check if element is within form context before interacting
            if not self._is_element_in_form_context(element, frame):
                context_str = "iframe" if frame else "page"
                print(f"⏭️ Skipping text field outside form context in {context_str}: {field.field_name}")
                return True
            
            # Try multiple approaches to interact with the field
            success = False
            
            # Approach 1: Focus and fill
            try:
                element.focus()
                time.sleep(0.3)
                element.fill(text if text else '')
                time.sleep(0.5)
                context_str = "iframe" if frame else "page"
                print(f"✅ Successfully typed '{text}' in {context_str} field using fill method")
                success = True
            except Exception as e:
                print(f"⚠️ Fill method failed: {e}")
            
            # Approach 2: Click and type if fill failed
            if not success:
                try:
                    element.click(force=True, timeout=5000)
                    time.sleep(0.3)
                    element.fill('')
                    time.sleep(0.2)
                    element.type(text, delay=50)
                    time.sleep(0.5)
                    context_str = "iframe" if frame else "page"
                    print(f"✅ Successfully typed '{text}' in {context_str} field using click and type")
                    success = True
                except Exception as e:
                    print(f"⚠️ Click and type method failed: {e}")
            
            # Approach 3: JavaScript approach if both failed
            if not success:
                try:
                    js_code = (
                        "(element, value) => {"
                        "element.focus();"
                        "element.value = '';"
                        "element.value = value;"
                        "element.dispatchEvent(new Event('input', { bubbles: true }));"
                        "element.dispatchEvent(new Event('change', { bubbles: true }));"
                        "}"
                    )
                    if frame:
                        frame.evaluate(js_code, element, text)
                    else:
                        self.page.evaluate(js_code, element, text)
                    time.sleep(0.5)
                    context_str = "iframe" if frame else "page"
                    print(f"✅ Successfully typed '{text}' in {context_str} field using JavaScript")
                    success = True
                except Exception as e:
                    print(f"⚠️ JavaScript method failed: {e}")
            
            return success
            
        except Exception as e:
            context_str = "iframe" if frame else "page"
            print(f"❌ Error clicking and typing in {context_str} field: {e}")
            return False


# Example usage and testing
if __name__ == "__main__":
    # This would be integrated with the main HybridBrowserBot
    print("🧪 Application Filler module loaded")
    print("This module should be imported and used with HybridBrowserBot") 