#!/usr/bin/env python3
"""
Application Filler

A comprehensive job application form filler that can handle various form types
and automatically fill them based on user preferences using gpt-4.1 analysis.
"""

# Removed incorrect import
import time
import json
from typing import List, Dict, Optional, Any, Tuple
from playwright.sync_api import Page
import openai
from pydantic import BaseModel
from enum import Enum
import traceback
import os

class FieldType(Enum):
    TEXT = "text"
    SELECT = "select"
    RADIO = "radio"
    CHECKBOX = "checkbox"
    UPLOAD = "upload"

class ApplicationField(BaseModel):
    field_order: int
    field_name: str
    field_type: FieldType
    field_selector: str
    field_is_visible: bool
    field_value: Optional[str | bool] = None
    field_options: Optional[List[str]] = None
    field_required: Optional[bool] = None
    field_placeholder: Optional[str] = None
    
class ApplicationFieldResponse(BaseModel):
    fields: List[ApplicationField]
    
class ApplicationStateResponse(BaseModel):
    submitted: bool
    completed: bool
    error_in_submission: bool
    verification_required: bool
    more_forms: bool

class ApplicationFiller:
    def __init__(self, page: Page, preferences: Dict[str, Any] = None):
        """
        Initialize the Application Filler
        
        Args:
            page: Playwright page object
            preferences: User preferences for filling forms
        """
        self.page = page
        self.preferences = preferences or {}
        self.max_form_iterations = 10  # Prevent infinite loops
        self.current_iteration = 0
        
    def fill_application(self) -> bool:
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
            return self.run_main_algorithm()
            
        except Exception as e:
            print(f"❌ Error in application filling: {e}")
            return False
        
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
    
    def run_main_algorithm(self) -> bool:
        """
        Main form filling algorithm that continues until completion
        
        Returns:
            bool: True if all forms were successfully filled
        """
        print("🎯 Starting main form filling algorithm...")
        
        self.current_iteration = 0
        
        while self.current_iteration < self.max_form_iterations:
            self.current_iteration += 1
            print(f"\n🔄 Form iteration {self.current_iteration}/{self.max_form_iterations}")
            
            try:
                # Step 0: Click accept cookies button
                self.click_accept_cookies_button()
                
                # Step 1: Detect iframes and determine context
                iframe_context = self.detect_and_handle_iframes()
                
                # Step 2: Find all form fields based on context
                if iframe_context['use_iframe_context']:
                    # Use unified field detection functions with frame context
                    frame = iframe_context['frame']
                    form_fields = self.find_all_form_inputs(frame)
                    form_fields['iframe_context'] = iframe_context
                else:
                    # Use unified field detection functions for main page
                    form_fields = self.find_all_form_inputs()
                
                print(f"📋 Found {form_fields['total_fields']} total inputs:")
                print(f"  - {len(form_fields['text_input_fields'])} text input fields")
                print(f"  - {len(form_fields['radios'])} radio button groups")
                print(f"  - {len(form_fields['checkboxes'])} checkboxes")
                print(f"  - {len(form_fields['selectors'])} select fields")
                print(f"  - {1 if form_fields['upload_button'] else 0} upload buttons")
                
                for field in form_fields['text_input_fields']:
                    print(f"Field: {field.field_name} - {field.field_value} ({field.field_options})")
                for field in form_fields['selectors']:
                    print(f"Field: {field.field_name} - {field.field_value} ({field.field_options})")
                for field in form_fields['radios']:
                    print(f"Field: {field.field_name} - {field.field_value} ({field.field_options})")
                for field in form_fields['checkboxes']:
                    print(f"Field: {field.field_name} - {field.field_value} ({field.field_options})")
                print(f"Field: {form_fields['upload_button'].field_name} - {form_fields['upload_button'].field_value} ({form_fields['upload_button'].field_options})")
                    
                # Step 3: Fill all form fields
                success = self.fill_all_form_inputs(form_fields)
                if not success:
                    print("❌ Failed to fill form inputs")
                    return False
                
                # Step 4: Find and click submit button
                submit_result = self.find_and_click_submit_button()
                if not submit_result:
                    print("❌ Failed to find or click submit input")
                    return False
                
                # Step 5: Take screenshot and analyze submission
                screenshot_path = f"form_submission_{self.current_iteration}.png"
                self.page.screenshot(path=screenshot_path, full_page=True)
                print(f"📸 Screenshot saved: {screenshot_path}")
                
                # Step 6: Analyze if form was submitted successfully
                submission_result = self.check_form_submission_with_gpt(screenshot_path)
                
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
                    print(f"Iframe frame: {iframe_frame}")
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
    
    def find_all_text_input_fields_in_iframe(self, iframe_context: Dict[str, Any]) -> List[ApplicationField]:
        """
        Find all text input fields in iframe
        """
        frame = iframe_context['frame']
        return self.find_all_text_input_fields(frame)
        
    def find_all_radio_fields_in_iframe(self, iframe_context: Dict[str, Any]) -> List[ApplicationField]:
        """
        Find all radio fields in iframe
        """
        frame = iframe_context['frame']
        return self.find_all_radio_fields(frame)
    
    def find_all_checkbox_fields_in_iframe(self, iframe_context: Dict[str, Any]) -> List[ApplicationField]:
        """
        Find all checkbox fields in iframe
        """
        frame = iframe_context['frame']
        return self.find_all_checkbox_fields(frame)
    
    def find_and_handle_all_select_fields_in_iframe(self, iframe_context: Dict[str, Any]) -> List[ApplicationField]:
        """
        Find all select fields in iframe using gpt-4.1
        """
        try:
            import openai
            client = openai.OpenAI()
            
            # Get the iframe frame
            frame = iframe_context['frame']
            
            # Get page content and structure from iframe
            page_content = frame.content()
            
            # Step 1: Use gpt-4.1 to find all the select fields
            print("🤖 Using gpt-4.1 to find select fields...")
            prompt = f"""
            Analyze this HTML and identify all the select/dropdown fields.
            
            Look for:
            1. A select element or a dropdown component or an element mimicking a select element
                - If there is no <select> element it is possible that there is a custom dropdown component or an element mimicking a select element
                    - If that is the case you need to find the element that can be clicked to open the dropdown
            2. Custom dropdown components (divs, buttons that open options)
            3. React-style select components
            4. Any clickable elements that show options when clicked
            5. The type of the input field is "select" and not "text" otherwise it is not a select field
            
            For each select field found, return:
            - field_name: The label of the field in the html
            - field_selector: CSS selector to find the field
            - field_type: "select"
            - field_value: null (we'll determine this after seeing options)
            - field_options: null (we'll populate this after clicking)
            
            Return a JSON object with ApplicationField objects.
            
            CRITICAL GUIDELINES:
            1. Find all select/dropdown fields related to the job application
            2. Use proper CSS selectors that target the ACTUAL select elements
            3. DO NOT include iframe selectors - only target the form elements themselves
            4. Examples of good selectors: "select[name='city']", "div[id='state-dropdown']", "[class*='select']"
            5. Examples of bad selectors: "iframe#grnhse_iframe", "iframe iframe select[name='city']"
            
            Preferences: {self.preferences}
            HTML of the page: {page_content}
            """
            
            response = client.responses.parse(
                model="gpt-4.1",
                input=[
                    {"role": "system", "content": "You are an expert at analyzing HTML and identifying select/dropdown fields. Return only valid JSON with CSS selectors."},
                    {"role": "user", "content": prompt}
                ],
                text_format=ApplicationFieldResponse
            )
            
            all_fields = response.output_parsed.fields
            select_fields: List[ApplicationField] = [field for field in all_fields if field.field_type == FieldType.SELECT]
            
            print(f"✅ gpt-4.1 found {len(select_fields)} select fields")
            
            # Step 2: Iteratively click each select field to see the options
            enhanced_select_fields = []
            
            for i, select_field in enumerate(select_fields):
                print(f"🔄 Processing select field {i+1}/{len(select_fields)}: {select_field.field_name}")
                
                try:
                    # Find the select field element
                    select_element = frame.locator(select_field.field_selector)
                    if not select_element:
                        print(f"⚠️ Could not find select element with selector: {select_field.field_selector}")
                        continue
                    
                    # Step 3: Take HTML snapshot before clicking
                    html_before_click = frame.content()
                    
                    # Click the select field to open options
                    print(f"🎯 Clicking select field: {select_field.field_selector}")
                    # frame.wait_for_selector(select_field.field_selector, state="visible")
                    # select_element.scroll_into_view_if_needed()
                    # time.sleep(0.5)
                    select_element.click()
                    
                    # Wait for options to appear
                    time.sleep(1.0)
                    
                    # Step 4: Take HTML snapshot after clicking to see options
                    html_after_click = frame.content()
                    
                    # Step 5: Use gpt-4.1 to find the select field and the options
                    options_prompt = f"""
                    Analyze these two HTML snapshots to find the select field and its options.
                    
                    HTML BEFORE clicking the select field:
                    {html_before_click}
                    
                    HTML AFTER clicking the select field (options should now be visible):
                    {html_after_click}
                    
                    The select field selector is: {select_field.field_selector}
                    The select field name is: {select_field.field_name}
                    
                    Instructions:
                    1. Compare the two HTML snapshots to identify what changed when the select field was clicked
                    2. Find all the available options that are now visible
                    3. Determine the best option value based on the user preferences: {self.preferences}
                    4. Return a JSON object with:
                       - field_name: The field name
                       - field_selector: The CSS selector for the field
                       - field_type: "select"
                       - field_value: The best option value to select from the list of options
                            - You are not allowed to use values that are not in the list of options
                       - field_options: Array of all available options
                    
                    Focus on finding options that are now visible in the second HTML snapshot that weren't visible in the first.
                    """
                    
                    options_response = client.responses.parse(
                        model="gpt-4.1",
                        input=[
                            {"role": "system", "content": "You are an expert at analyzing HTML changes and identifying dropdown options. Return only valid JSON."},
                            {"role": "user", "content": options_prompt}
                        ],
                        text_format=ApplicationFieldResponse
                    )
                    
                    # Get the enhanced field with options
                    enhanced_field = options_response.output_parsed.fields[0] if options_response.output_parsed.fields else select_field
                    
                    print(f"✅ Found {len(enhanced_field.field_options) if enhanced_field.field_options else 0} options for {enhanced_field.field_name}")
                    print(f"   Best value: {enhanced_field.field_value}")
                    
                    enhanced_select_fields.append(enhanced_field)
                    frame.get_by_role("option", name=enhanced_field.field_value).click(force=True, timeout=5000)
                    print(f"🔄 Clicked {enhanced_field.field_name}")
                    # Close the dropdown by clicking outside or pressing Escape
                    try:
                        # Try clicking outside the dropdown
                        frame.click("body", position={"x": 0, "y": 0})
                    except:
                        # Try pressing Escape
                        frame.keyboard.press("Escape")
                    
                    time.sleep(0.5)  # Wait for dropdown to close
                    
                except Exception as e:
                    print(f"❌ Error processing select field {select_field.field_name}: {e}")
                    # Add the original field without options
                    enhanced_select_fields.append(select_field)
                    continue
            
            print(f"✅ Successfully processed {len(enhanced_select_fields)} select fields with options")
            return enhanced_select_fields
            
        except Exception as e:
            print(f"❌ Error in find_and_handle_all_select_fields_in_iframe: {e}")
            return []
    
    def find_upload_file_button_in_iframe(self, iframe_context: Dict[str, Any]) -> ApplicationField:
        """
        Find all upload file buttons in iframe
        """
        frame = iframe_context['frame']
        return self.find_upload_file_button(frame)
    
    def fill_all_form_inputs(self, form_fields: Dict[str, List[ApplicationField]]) -> bool: 
        """Fill all form inputs using unified functions"""
        try:
            # Check if we're working with iframe context
            iframe_context = form_fields.get('iframe_context')
            frame = iframe_context['frame'] if iframe_context else None
            
            context_str = "iframe" if frame else "main page"
            print(f"🎯 Filling form fields in {context_str} context...")
            
            # Fill text input fields
            for field in form_fields.get('text_input_fields', []):
                self.click_and_type_in_field(field, field.field_value, frame)
            
            # Fill select fields
            for field in form_fields.get('selectors', []):
                self.handle_select_field(field, frame)
            
            # Fill radio buttons
            for field in form_fields.get('radios', []):
                self.click_radio_button(field, frame)
            
            # Fill checkboxes
            for field in form_fields.get('checkboxes', []):
                self.click_checkbox(field, frame)
            
            # Fill upload buttons
            for field in form_fields.get('upload_button', []):
                self.handle_file_upload(field, frame)
            
            return True
        except Exception as e:
            print(f"❌ Error filling form inputs: {e}")
            traceback.print_exc()
            return False
    
    def click_and_type_in_field_iframe(self, field: ApplicationField, text: str, frame) -> bool:
        """Click on a field and type text using Playwright within iframe context"""
        return self.click_and_type_in_field(field, text, frame)
    
    def click_radio_button_iframe(self, radio_button: ApplicationField, frame) -> bool:
        """Click on a radio button within iframe context"""
        return self.click_radio_button(radio_button, frame)
        
    def click_checkbox_iframe(self, checkbox: ApplicationField, frame) -> bool:
        """Click on a checkbox within iframe context"""
        return self.click_checkbox(checkbox, frame)
        
    def click_upload_button_iframe(self, upload_button: ApplicationField, frame) -> bool:
        """Click on an upload button within iframe context"""
        return self.click_upload_button(upload_button, frame)
        
    def handle_select_field_iframe(self, select_field: ApplicationField, frame) -> bool:
        """Handle a select field within iframe context"""
        return self.handle_select_field(select_field, frame)
    
    def handle_select_field(self, select_field: ApplicationField, frame=None) -> bool:
        """Handle select field in page or iframe context"""
        try:
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
            
            # Check if element is within form context before interacting
            if not self._is_element_in_form_context(element, frame):
                context_str = "iframe" if frame else "page"
                print(f"⏭️ Skipping select field outside form context in {context_str}: {select_field.field_name}")
                return True
            
            # Check if this is a system or preference field that should be skipped
            if self._is_system_or_preference_field(select_field):
                context_str = "iframe" if frame else "page"
                print(f"⏭️ Skipping system/preference select field in {context_str}: {select_field.field_name}")
                return True
            
            select_value: bool = select_field.field_value
            if not select_value:
                context_str = "iframe" if frame else "page"
                print(f"⏭️ Skipping select field with false value in {context_str}: {select_field.field_name}")
                return True
            
            # Try to handle as standard select first
            try:
                # Check if already selected
                if self._check_select_already_selected(element, str(select_value), frame):
                    context_str = "iframe" if frame else "page"
                    print(f"✅ Select field already has correct value in {context_str}: {select_field.field_name}")
                    return True
                
                # Try standard select handling
                element.select_option(value=str(select_value))
                context_str = "iframe" if frame else "page"
                print(f"✅ Successfully selected option in {context_str}: {select_field.field_name}")
                return True
                
            except Exception as e:
                print(f"⚠️ Standard select handling failed, trying custom dropdown: {e}")
                # Fallback to custom dropdown handling
                return self._handle_custom_dropdown(select_field, frame)
                
        except Exception as e:
            context_str = "iframe" if frame else "page"
            print(f"❌ Error handling select field in {context_str}: {e}")
            return False

    def _handle_custom_dropdown_iframe(self, select_field: ApplicationField, frame) -> bool:
        """Handle custom dropdown components within iframe context"""
        return self._handle_custom_dropdown(select_field, frame)
    
    def _handle_custom_dropdown(self, select_field: ApplicationField, frame=None) -> bool:
        """Handle custom dropdown components in main page context using gpt-4.1"""
        try:
            select_value: str = select_field.field_value
            print(f"🤖 Using gpt-4.1 to handle dropdown in main page for value: {select_value}")
            
            # Step 1: Take HTML snapshot to find the dropdown field
            print("📸 Taking HTML snapshot to find dropdown field...")
            page_html = self.page.content()
            
            # Step 2: Use gpt-4.1 to find the dropdown field
            dropdown_info = self._find_dropdown_field_with_gpt(page_html, select_field.field_selector, select_value, "main page")
            if not dropdown_info:
                print(f"❌ gpt-4.1 could not find dropdown field")
                return False
            
            # Step 3: Click the dropdown field
            print(f"🎯 Clicking dropdown field: {dropdown_info['selector']}")
            try:
                dropdown_element = self.page.locator(dropdown_info['selector'])
                if dropdown_element:
                    dropdown_element.scroll_into_view_if_needed()
                    time.sleep(0.5)
                    dropdown_element.click(force=True, timeout=5000)
                    print("✅ Clicked dropdown field")
                else:
                    print(f"❌ Could not find dropdown element with selector: {dropdown_info['selector']}")
                    return False
            except Exception as e:
                print(f"❌ Error clicking dropdown field: {e}")
                return False
            
            # Step 4: Take another HTML snapshot to see the options
            print("📸 Taking HTML snapshot to see dropdown options...")
            time.sleep(1.0)  # Wait for dropdown to open
            page_html_after_click = self.page.content()
            
            # Step 5: Use gpt-4.1 to click the correct option
            option_info = self._find_and_click_option_with_gpt(page_html_after_click, select_value, self.page)
            if option_info:
                print(f"✅ Successfully selected option: {select_value}")
                time.sleep(0.5)  # Wait for selection to register
                return True
            else:
                print(f"❌ Could not find or click option: {select_value}")
                return False
                
        except Exception as e:
            print(f"❌ Error handling custom dropdown in main page with gpt-4.1: {e}")
            return False
    
    def _find_dropdown_field_with_gpt(self, page_html: str, original_selector: str, target_value: str, context: str) -> Optional[Dict[str, Any]]:
        """Use gpt-4.1 to find the dropdown field from HTML snapshot"""
        try:
            import openai
            import json
            
            prompt = f"""
            Analyze this HTML and find the dropdown field that needs to be clicked to select the value "{target_value}".
            
            The original selector provided was: {original_selector}
            Context: {context}
            
            Instructions:
            1. Look for the dropdown field that contains or will show the option "{target_value}"
            2. This could be a <select> element, a custom dropdown div, or any clickable element that opens options
            3. Return a JSON object with:
               - selector: A CSS selector that will find the dropdown field
               - element_type: The type of element (select, div, button, etc.)
               - confidence: 0.0-1.0 confidence level
            
            If you cannot find the dropdown field, return null.
            
            HTML content:
            {page_html}
            """
            
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing HTML and finding dropdown fields. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
            )
            
            result = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                if result and result.lower() != "null":
                    # Clean up the result
                    if result.startswith("```json"):
                        result = result[7:]
                    if result.endswith("```"):
                        result = result[:-3]
                    
                    dropdown_data = json.loads(result.strip())
                    
                    if 'selector' in dropdown_data:
                        print(f"✅ gpt-4.1 found dropdown field: {dropdown_data['selector']}")
                        return dropdown_data
                    else:
                        print("⚠️ GPT response missing selector field")
                        return None
                else:
                    return None
                    
            except json.JSONDecodeError as e:
                print(f"⚠️ Could not parse GPT response as JSON: {e}")
                return None
                
        except Exception as e:
            print(f"❌ Error in gpt-4.1 dropdown field detection: {e}")
            return None
    
    def _find_and_click_option_with_gpt(self, page_html: str, target_value: str, frame) -> Optional[Dict[str, Any]]:
        """Use gpt-4.1 to find and click the correct option from HTML snapshot"""
        try:
            import openai
            
            prompt = f"""
            Analyze this HTML and return the selector that will find the option that matches "{target_value}".
            
            The dropdown is now open and showing options. Find the option that should be clicked to select "{target_value}".
            
            Look for dropdown options that are now visible and find the css selector that will find the option element exactly.
                This could be with the text of the option or with the value of the option or with the id of the option or with the class of the option.
                The selector must be exact and must not be a parent or a child of the option element.
            
            You must only return the selector that will find the option element.
            You must not return any other text.
            
            Example of a good response:
            "div[class='dropdown-option']"
            
            
            HTML content (dropdown is open):
            {page_html}
            """
            
            client = openai.OpenAI()
            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing HTML and finding dropdown options. Return only the selector that will find the option element."},
                    {"role": "user", "content": prompt}
                ],
            )
            
            result = response.choices[0].message.content.strip()
            print(result)
        
            option_element = frame.locator(result)
            if option_element:
                option_element.click(force=True, timeout=5000)
                print(f"✅ Clicked option: {result}")
                return result
            else:
                print(f"❌ Could not find option element with selector: {result}")
                return None
            
        except Exception as e:
            print(f"❌ Error in gpt-4.1 option detection: {e}")
            return None
    
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
    
    def _is_system_or_preference_field(self, select_field: ApplicationField) -> bool:
        """
        Detect system or preference fields that should typically be skipped
        """
        field_name = select_field.field_name.lower() if select_field.field_name else ""
        field_selector = select_field.field_selector.lower() if select_field.field_selector else ""
        
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
    
    # Removed _get_enhanced_option_selectors and _is_valid_option_match methods
    # These are no longer needed since we use gpt-4.1 for intelligent dropdown handling
    
    def _get_file_upload_button_selectors(self, file_input_id: str = None, file_input_name: str = None) -> list:
        """
        Generate comprehensive list of selectors for file upload buttons
        """
        selectors = [
            # Generic upload buttons
            'button:has-text("Attach")',
            'button:has-text("Upload")',
            'button:has-text("Browse")',
            'button:has-text("Choose File")',
            'button:has-text("Select File")',
            'button:has-text("Add File")',
            'button:has-text("Choose")',
            '.btn:has-text("Attach")',
            '.btn:has-text("Upload")',
            '.btn:has-text("Browse")',
            
            # Class-based selectors
            '.file-upload button',
            '.upload-button',
            '.file-input-button',
            '.attach-button',
            '[class*="upload"] button',
            '[class*="file"] button',
            '[class*="attach"] button',
            
            # Data attribute selectors
            'button[data-testid*="upload"]',
            'button[data-testid*="file"]',
            'button[data-testid*="attach"]',
            'button[data-testid*="resume"]',
            'button[data-testid*="cv"]',
        ]
        
        # Add specific selectors based on input attributes
        if file_input_id:
            selectors.extend([
                f'button[data-for="{file_input_id}"]',
                f'button[for="{file_input_id}"]',
                f'[data-target="#{file_input_id}"]',
            ])
        
        if file_input_name:
            selectors.extend([
                f'button[data-name="{file_input_name}"]',
                f'button:has-text("{file_input_name}")',
            ])
        
        return selectors
    
    def _is_clickable_element(self, element) -> bool:
        """
        Check if an element is clickable and visible
        """
        try:
            if not element:
                return False
            
            # Check if element is visible
            if not element.is_visible():
                return False
            
            # Check if element is enabled
            if not element.is_enabled():
                return False
            
            return True
        except:
            return False
    
    def _safe_click_element(self, element) -> bool:
        """
        Safely click an element with multiple fallback strategies
        """
        try:
            if not element:
                return False
            
            # Strategy 1: Scroll into view and direct click
            try:
                element.scroll_into_view_if_needed()
                time.sleep(0.3)
                element.click(force=True, timeout=5000)
                print(f"✅ Element clicked successfully")
                return True
            except Exception as e:
                print(f"⚠️ Direct click failed: {e}")
            
            # Strategy 2: JavaScript click
            try:
                self.page.evaluate("(el) => el.click()", element)
                print(f"✅ Element clicked with JavaScript")
                return True
            except Exception as e:
                print(f"⚠️ JavaScript click failed: {e}")
            
            # Strategy 3: Dispatch click event
            try:
                element.dispatch_event("click")
                print(f"✅ Click event dispatched")
                return True
            except Exception as e:
                print(f"⚠️ Event dispatch failed: {e}")
            
            # Strategy 4: Try waiting and retrying
            try:
                time.sleep(1)
                element.click(force=True, timeout=2000)
                print(f"✅ Element clicked after retry")
                return True
            except Exception as e:
                print(f"⚠️ Retry click failed: {e}")
            
            return False
            
        except Exception as e:
            print(f"❌ Safe click failed: {e}")
            return False
    
    def _handle_field_error(self, field: ApplicationField, error: Exception, context: str = "") -> bool:
        """
        Handle field processing errors with intelligent recovery
        """
        print(f"⚠️ Error processing {field.field_type.value} field '{field.field_name}': {error}")
        
        # Log the error for debugging
        if context:
            print(f"📍 Context: {context}")
        
        # For select fields, try a simplified approach
        if field.field_type == FieldType.SELECT:
            try:
                print(f"🔄 Attempting simplified select handling for {field.field_name}")
                # Just return True to skip problematic selects gracefully
                return True
            except:
                pass
        
        # For text fields, try a basic approach
        if field.field_type == FieldType.TEXT:
            try:
                print(f"🔄 Attempting simplified text input for {field.field_name}")
                element = self.page.locator(field.field_selector)
                if element:
                    element.fill(field.field_value)
                    return True
            except:
                pass
        
        # For upload fields, try direct file path setting
        if field.field_type == FieldType.UPLOAD:
            try:
                print(f"🔄 Attempting direct file upload for {field.field_name}")
                file_input = self.page.locator('input[type="file"]')
                if file_input and hasattr(field, 'file_path'):
                    file_input.set_input_files(field.file_path)
                    return True
            except:
                pass
        
        # Return True to continue with other fields rather than failing completely
        print(f"⏭️ Skipping problematic field: {field.field_name}")
        return True
    
    def click_and_type_in_field(self, field: ApplicationField, text: str, frame=None) -> bool:
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
            
            # Check if element is within form context before interacting
            if not self._is_element_in_form_context(element, frame):
                context_str = "iframe" if frame else "page"
                print(f"⏭️ Skipping text field outside form context in {context_str}: {field.field_name}")
                return True
            
            # Scroll element into view first
            try:
                element.scroll_into_view_if_needed()
                time.sleep(0.5)
            except Exception as e:
                print(f"⚠️ Scroll failed, continuing anyway: {e}")
            
            # Try multiple approaches to interact with the field
            success = False
            
            # Approach 1: Focus and fill
            try:
                element.focus()
                time.sleep(0.3)
                element.fill('')
                time.sleep(0.2)
                element.fill(text)
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
                    js_code = f"""
                        (element) => {{
                            element.focus();
                            element.value = '';
                            element.value = '{text}';
                            element.dispatchEvent(new Event('input', {{ bubbles: true }}));
                            element.dispatchEvent(new Event('change', {{ bubbles: true }}));
                        }}
                    """
                    if frame:
                        frame.evaluate(js_code, element)
                    else:
                        self.page.evaluate(js_code, element)
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

    def click_radio_button(self, radio_button: ApplicationField, frame=None) -> bool:
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

    def click_checkbox(self, checkbox: ApplicationField, frame=None) -> bool:
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

    def click_upload_button(self, upload_button: ApplicationField, frame=None) -> bool:
        """Click on an upload button in page or iframe context"""
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
                
            upload_value: bool = upload_button.field_value
            if upload_value:
                element.click()
                context_str = "iframe" if frame else "page"
                print(f"✅ Clicked upload button in {context_str}")
            return True
        except Exception as e:
            context_str = "iframe" if frame else "page"
            print(f"❌ Error clicking upload button in {context_str}: {e}")
            return False

    def handle_file_upload_iframe(self, upload_button: ApplicationField, frame, times_tried=0) -> bool:
        """
        Handle file upload in iframe context
        
        Args:
            upload_button: Upload button field information
            frame: Iframe frame object
            
        Returns:
            bool: True if file upload was handled successfully
        """
        return self.handle_file_upload(upload_button, frame, times_tried)
            
    
    def handle_file_upload(self, upload_button: ApplicationField, frame=None, times_tried=0) -> bool:
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
            
            upload_value: bool = upload_button.field_value
            if not upload_value:
                context_str = "iframe" if frame else "page"
                print(f"⏭️ Skipping upload button with false value in {context_str}: {upload_button.field_name}")
                return True
            
            try:
                element.click()
                
                context_str = "iframe" if frame else "page"
                print(f"✅ Successfully triggered file dialog in {context_str}")
                return True
            except Exception as e:
                print(f"⚠️ File dialog triggering failed: {e}")
            
            # If all methods failed and we haven't tried too many times, retry
            if times_tried < 3:
                print(f"🔄 Retrying file upload (attempt {times_tried + 1}/3)")
                time.sleep(1)
                return self.handle_file_upload(upload_button, frame, times_tried + 1)
            
            context_str = "iframe" if frame else "page"
            print(f"❌ Failed to upload file in {context_str} after {times_tried + 1} attempts")
            
            return False
            
        except Exception as e:
            context_str = "iframe" if frame else "page"
            print(f"❌ Error handling file upload in {context_str}: {e}")
            return False

    def _trigger_file_dialog(self, file_input, frame=None) -> bool:
        """Trigger file dialog in page or iframe context"""
        try:
            # Try clicking the file input
            time.sleep(0.5)
            time.sleep(0.5)
            return True
            
        except Exception as e:
            print(f"⚠️ Error triggering file dialog: {e}")
            return False

    def _set_file_path_directly(self, file_input, file_path: str, frame=None) -> bool:
        """Set file path directly in page or iframe context"""
        try:
            # Try to set file path using JavaScript
            js_code = f"""
                (element) => {{
                    element.style.display = 'block';
                    element.style.visibility = 'visible';
                    element.style.opacity = '1';
                    element.style.position = 'static';
                    
                    // Create a new FileList-like object
                    const file = new File([''], '{os.path.basename(file_path)}', {{ type: 'application/octet-stream' }});
                    const dataTransfer = new DataTransfer();
                    dataTransfer.items.add(file);
                    element.files = dataTransfer.files;
                    
                    // Trigger change event
                    element.dispatchEvent(new Event('change', {{ bubbles: true }}));
                    element.dispatchEvent(new Event('input', {{ bubbles: true }}));
                }}
            """
            
            if frame:
                frame.evaluate(js_code, file_input)
            else:
                self.page.evaluate(js_code, file_input)
            
            time.sleep(0.5)
            return True
            
        except Exception as e:
            print(f"⚠️ Error setting file path directly: {e}")
            return False

    def find_submit_button_with_gpt(self, iframe_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Use gpt-4.1 to contextually find submit buttons on the page or in iframe
        
        Args:
            iframe_context: Iframe context if searching within an iframe
            
        Returns:
            Dict with submit button information and action
        """
        try:
            print("🤖 Using gpt-4.1 to find submit button contextually...")
            
            # Get the appropriate content and frame
            if iframe_context:
                frame = iframe_context['frame']
                page_content = frame.content()
                context_name = f"iframe {iframe_context['index']+1}"
            else:
                frame = self.page
                page_content = self.page.content()
                context_name = "main page"
            
            import openai
            client = openai.OpenAI()
            
            prompt = f"""
            Analyze this HTML and find the submit/continue button for a job application form.
            
            Look for buttons that would submit or continue the application process.
            Common submit button indicators:
            - Text like "Submit", "Continue", "Next", "Apply", "Send", "Save"
            - Buttons with submit-related classes or IDs
            - Form submit buttons
            - Buttons that appear to advance the application process
            
            Return a JSON object with:
            {{
                "submit_button_found": true/false,
                "button_selector": "CSS selector for the button",
                "button_text": "Text content of the button",
                "button_type": "submit/button/input",
                "confidence": 0.0-1.0,
                "reasoning": "Why you think this is the submit button"
            }}
            
            If no submit button is found, return:
            {{
                "submit_button_found": false,
                "button_selector": null,
                "button_text": null,
                "button_type": null,
                "confidence": 0.0,
                "reasoning": "Why no submit button was found"
            }}
            
            HTML content from {context_name}:
            {page_content}
            """
            
            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing HTML and identifying submit buttons for job application forms. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
            )
            
            result = response.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                analysis = json.loads(result)
                submit_button_found = analysis.get('submit_button_found', False)
                button_selector = analysis.get('button_selector')
                button_text = analysis.get('button_text')
                confidence = analysis.get('confidence', 0.0)
                reasoning = analysis.get('reasoning', '')
                
                if submit_button_found and button_selector and confidence > 0.5:
                    print(f"✅ gpt-4.1 found submit button: '{button_text}' (confidence: {confidence:.2f})")
                    print(f"   Selector: {button_selector}")
                    print(f"   Reasoning: {reasoning}")
                    
                    # Try to find the button using the selector
                    try:
                        button_element = frame.locator(button_selector)
                        print(f"Button element: {button_element}")
                        if button_element and button_element.is_enabled():
                            return {
                                'found': True,
                                'element': button_element,
                                'selector': button_selector,
                                'text': button_text,
                                'confidence': confidence,
                                'reasoning': reasoning,
                                'context': context_name
                            }
                        else:
                            print(f"⚠️ Button found by GPT but not accessible: {button_selector}")
                    except Exception as e:
                        print(f"⚠️ Error accessing button with selector {button_selector}: {e}")
                
                else:
                    print(f"ℹ️ gpt-4.1 analysis: {reasoning}")
                    
            except json.JSONDecodeError:
                print(f"⚠️ Could not parse GPT response: {result}")
            
            return {
                'found': False,
                'element': None,
                'selector': None,
                'text': None,
                'confidence': 0.0,
                'reasoning': 'GPT analysis failed or no button found',
                'context': context_name
            }
            
        except Exception as e:
            print(f"❌ Error in GPT submit button detection: {e}")
            return {
                'found': False,
                'element': None,
                'selector': None,
                'text': None,
                'confidence': 0.0,
                'reasoning': f'Error: {e}',
                'context': context_name if 'context_name' in locals() else 'unknown'
            }

    def find_and_click_submit_button(self) -> bool:
        """
        Find and click the submit/continue button using gpt-4.1 contextual analysis
        
        Returns:
            bool: True if button was found and clicked successfully
        """
        try:
            print("🔍 Looking for submit/continue button using gpt-4.1...")
            
            # First, check if there are any iframes on the page
            iframes = self.page.query_selector_all('iframe')
            visible_iframes = [iframe for iframe in iframes if iframe.is_visible()]
            
            if visible_iframes:
                print(f"📋 Found {len(visible_iframes)} visible iframes - checking if submit button is in iframe...")
                
                # Check iframes first for submit button
                for i, iframe in enumerate(visible_iframes):
                    try:
                        iframe_frame = iframe.content_frame()
                        if not iframe_frame:
                            continue
                        
                        # Check if iframe contains form elements
                        form_elements = iframe_frame.query_selector_all('input, select, textarea')
                        if not form_elements:
                            continue
                        
                        print(f"🔍 Checking iframe {i+1} for submit button...")
                        
                        # Use gpt-4.1 to find submit button in iframe
                        iframe_context = {
                            'index': i,
                            'iframe': iframe,
                            'frame': iframe_frame,
                            'form_count': len(form_elements)
                        }
                        
                        gpt_result = self.find_submit_button_with_gpt(iframe_context)
                        
                        if gpt_result['found']:
                            print(f"🎯 Found submit button in iframe {i+1}: '{gpt_result['text']}'")
                            return self._click_submit_button(gpt_result)
                            
                    except Exception as e:
                        print(f"⚠️ Error checking iframe {i+1} for submit button: {e}")
                        continue
                
                # If not found in iframes, check main page
                print("🔍 Submit button not found in iframes, checking main page...")
                gpt_result = self.find_submit_button_with_gpt()
                
                if gpt_result['found']:
                    print(f"🎯 Found submit button in main page: '{gpt_result['text']}'")
                    return self._click_submit_button(gpt_result)
            else:
                print("ℹ️ No iframes found - checking main page for submit button...")
                # No iframes, check main page directly
                gpt_result = self.find_submit_button_with_gpt()
                
                if gpt_result['found']:
                    print(f"🎯 Found submit button in main page: '{gpt_result['text']}'")
                    return self._click_submit_button(gpt_result)
            
            # Fallback to traditional selectors if gpt-4.1 didn't find anything
            print("🔄 gpt-4.1 didn't find submit button, trying traditional selectors...")
            return self._find_and_click_submit_button_fallback()
            
        except Exception as e:
            print(f"❌ Error in GPT submit button detection: {e}")
            # Fallback to traditional method
            return self._find_and_click_submit_button_fallback()
    
    def _click_submit_button(self, gpt_result: Dict[str, Any]) -> bool:
        """
        Click the submit button found by gpt-4.1
        
        Args:
            gpt_result: Result from gpt-4.1 submit button detection
            
        Returns:
            bool: True if button was clicked successfully
        """
        try:
            button_element = gpt_result['element']
            button_text = gpt_result['text']
            context = gpt_result['context']
            
            print(f"🎯 Clicking submit button '{button_text}' in {context}...")
            
            # Scroll button into view
            button_element.scroll_into_view_if_needed()
            time.sleep(0.5)
            
            # Click the button
            button_element.click()
            
            # Wait for form submission
            time.sleep(3)
            
            print(f"✅ Successfully clicked submit button '{button_text}' in {context}")
            return True
            
        except Exception as e:
            print(f"❌ Error clicking submit button: {e}")
            print("🎯 No submit button found, finding next button...")
            return self.find_and_click_next_button()
            return False
    
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
        next_button = self.page.locator("button:has-text(/^(Next|Continue)$/i)")
        if next_button.is_visible():
            next_button.click()
            time.sleep(3)
            print("✅ Clicked next button")
            return True
        return False
    
    def check_form_submission_with_gpt(self, screenshot_path: str) -> ApplicationStateResponse:
        """
        Use gpt-4.1 to analyze if the form was successfully submitted
        
        Args:
            screenshot_path: Path to the screenshot file
            
        Returns:
            Dict with 'submitted' and 'completed' status
        """
        try:
            # For now, use page content analysis instead of image analysis
            # In a real implementation, you could use GPT-4 Vision API
            
            page_content = self.page.content()
            current_url = self.page.url
            
            client = openai.OpenAI()
            
            prompt = f"""
            Analyze this job application page to determine if a form was successfully submitted and if the application process is complete.
            
            CURRENT URL: {current_url}
            
            PAGE CONTENT:
            {page_content}
            
            INSTRUCTIONS:
            1. Look for indicators that a form was successfully submitted:
               - Success messages ("Thank you", "Application submitted", "Form submitted", etc)
               - Confirmation pages
               - Progress indicators showing advancement
               - New form sections appearing
               - URL changes indicating progression
            
            2. Determine if the application process is complete:
               - Final confirmation messages
               - "Application complete" or similar messages
               - No more forms or steps indicated
               - Thank you page or final confirmation
            
            Return only the JSON, nothing else.
            """
            
            response = client.responses.parse(
                model="gpt-4.1",
                input=[
                    {"role": "system", "content": "You are an expert at analyzing job application pages. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                text_format=ApplicationStateResponse
            )
            
            result: ApplicationStateResponse = response.output_parsed
            
            # Parse JSON response
            try:
                print(f"🤖 GPT analysis - Submitted: {result.submitted}, Completed: {result.completed}, Verification required: {result.verification_required}, More forms: {result.more_forms}, Error in submission: {result.error_in_submission}")
                return result
                
            except json.JSONDecodeError:
                print(f"⚠️ Could not parse GPT response: {result}")
                return ApplicationStateResponse(submitted=False, completed=False, verification_required=False, more_forms=False, error_in_submission=False)
                
        except Exception as e:
            print(f"❌ Error in GPT submission analysis: {e}")
            return ApplicationStateResponse(submitted=False, completed=False, verification_required=False, more_forms=False, error_in_submission=False)

    def find_all_text_input_fields(self, frame=None) -> List[ApplicationField]:
        """
        Find all text input fields in page or iframe context
        
        Args:
            frame: Optional iframe frame context. If None, uses main page.
        """
        try:
            import openai
            client = openai.OpenAI()
            
            # Get page content from frame or main page
            if frame:
                page_content = frame.content()
            else:
                page_content = self.page.content()
            
            # Use gpt-4.1 to analyze the page and identify form fields
            prompt = f"""
            Analyze this HTML and identify all the text input fields.
            Also fill in the values for the input using the preferences.
            For each input you need to fill in the value of the input.
            Ensure that the input is not used for a dropdown or a radio button or a checkbox.
            The input should be contextually used to enter a value.
            Some text inputs are used for a dropdown or a radio button or a checkbox, do not include them.
            
            For a text input the possible values are a string
            
            Return a JSON object with ApplicationField objects.
            
            CRITICAL GUIDELINES:
            1. Find all the form inputs related to the job application.
            2. Use proper CSS selectors that target the ACTUAL form elements (input, select, textarea, etc.).
            3. DO NOT include iframe selectors like "iframe#grnhse_iframe" - only target the form elements themselves.
            4. If form elements are inside an iframe, use selectors that would work within that iframe context.
            5. Examples of good selectors: "input[name='name']", "select[id='city']", "input[type='email']"
            6. Examples of bad selectors: "iframe#grnhse_iframe", "iframe iframe input[name='name']"
            
            Preferences: {self.preferences}
            HTML of the page: {page_content}
            """
            
            response = client.responses.parse(
                model="gpt-4.1",
                input=[
                    {"role": "system", "content": "You are an expert at analyzing HTML and identifying form inputs. Return only valid JSON with CSS selectors."},
                    {"role": "user", "content": prompt}
                ],
                text_format=ApplicationFieldResponse
            )
            
            all_fields = response.output_parsed.fields
            
            all_fields = [field for field in all_fields if field.field_is_visible]
            
            # Parse the JSON response
            text_input_fields: List[ApplicationField] = [field for field in all_fields if field.field_type == FieldType.TEXT]
            
            context_str = "iframe" if frame else "page"
            print(f"✅ gpt-4.1 found {len(text_input_fields)} text input fields in {context_str}")
            
            return text_input_fields
            
        except Exception as e:
            print(f"❌ Error in find_all_text_input_fields: {e}")
            return []

    def find_all_radio_fields(self, frame=None) -> List[ApplicationField]:
        """
        Find all radio fields in page or iframe context
        
        Args:
            frame: Optional iframe frame context. If None, uses main page.
        """
        try:
            import openai
            client = openai.OpenAI()
            
            # Get page content from frame or main page
            if frame:
                page_content = frame.content()
            else:
                page_content = self.page.content()
            
            # Use gpt-4.1 to analyze the page and identify form fields
            prompt = f"""
            Analyze this HTML and identify all the radio fields.
            Also fill in the values for the input using the preferences.
            For each input you need to fill in the value of the input.
            
            For a radio input the possible values are a boolean (true/false)
            
            Return a JSON object with ApplicationField objects.
            
            CRITICAL GUIDELINES:
            1. Find all the form inputs related to the job application.
            2. Use proper CSS selectors that target the ACTUAL form elements (input, select, textarea, etc.).
            3. DO NOT include iframe selectors like "iframe#grnhse_iframe" - only target the form elements themselves.
            4. If form elements are inside an iframe, use selectors that would work within that iframe context.
            5. Examples of good selectors: "input[name='name']", "select[id='city']", "input[type='email']"
            6. Examples of bad selectors: "iframe#grnhse_iframe", "iframe iframe input[name='name']"
            
            Preferences: {self.preferences}
            HTML of the page: {page_content}
            """
            
            response = client.responses.parse(
                model="gpt-4.1",
                input=[
                    {"role": "system", "content": "You are an expert at analyzing HTML and identifying form inputs. Return only valid JSON with CSS selectors."},
                    {"role": "user", "content": prompt}
                ],
                text_format=ApplicationFieldResponse
            )
            
            all_fields = response.output_parsed.fields
            
            all_fields = [field for field in all_fields if field.field_is_visible]
            
            context_str = "iframe" if frame else "page"
            
            # Parse the JSON response
            radio_fields: List[ApplicationField] = [field for field in all_fields if field.field_type == FieldType.RADIO]
            
            print(f"✅ gpt-4.1 found {len(radio_fields)} radio fields in {context_str}")
            
            return radio_fields
            
        except Exception as e:
            print(f"❌ Error in find_all_radio_fields: {e}")
            return []

    def find_all_checkbox_fields(self, frame=None) -> List[ApplicationField]:
        """
        Find all checkbox fields in page or iframe context
        
        Args:
            frame: Optional iframe frame context. If None, uses main page.
        """
        try:
            import openai
            client = openai.OpenAI()
            
            # Get page content from frame or main page
            if frame:
                page_content = frame.content()
            else:
                page_content = self.page.content()
            
            # Use gpt-4.1 to analyze the page and identify form fields
            prompt = f"""
            Analyze this HTML and identify all the checkbox fields.
            Also fill in the values for the input using the preferences.
            For each input you need to fill in the value of the input.
            
            For a checkbox input the possible values are a boolean (true/false)
            
            Return a JSON object with ApplicationField objects.
            
            CRITICAL GUIDELINES:
            1. Find all the form inputs related to the job application.
            2. Use proper CSS selectors that target the ACTUAL form elements (input, select, textarea, etc.).
            3. DO NOT include iframe selectors like "iframe#grnhse_iframe" - only target the form elements themselves.
            4. If form elements are inside an iframe, use selectors that would work within that iframe context.
            5. Examples of good selectors: "input[name='name']", "select[id='city']", "input[type='email']"
            6. Examples of bad selectors: "iframe#grnhse_iframe", "iframe iframe input[name='name']"
            
            Preferences: {self.preferences}
            HTML of the page: {page_content}
            """
            
            response = client.responses.parse(
                model="gpt-4.1",
                input=[
                    {"role": "system", "content": "You are an expert at analyzing HTML and identifying form inputs. Return only valid JSON with CSS selectors."},
                    {"role": "user", "content": prompt}
                ],
                text_format=ApplicationFieldResponse
            )
            
            all_fields = response.output_parsed.fields
            
            all_fields = [field for field in all_fields if field.field_is_visible]
            
            context_str = "iframe" if frame else "page"
            # Parse the JSON response
            checkbox_fields: List[ApplicationField] = [field for field in all_fields if field.field_type == FieldType.CHECKBOX]
            
            print(f"✅ gpt-4.1 found {len(checkbox_fields)} checkbox fields in {context_str}")
            
            return checkbox_fields
            
        except Exception as e:
            print(f"❌ Error in find_all_checkbox_fields: {e}")
            return []

    def find_and_handle_all_select_fields(self, frame=None) -> List[ApplicationField]:
        """
        Find all select fields in page or iframe context
        
        Args:
            frame: Optional iframe frame context. If None, uses main page.
        """
        try:
            import openai
            client = openai.OpenAI()
            
            # Get page content from frame or main page
            if frame:
                page_content = frame.content()
            else:
                page_content = self.page.content()
            
            # Use gpt-4.1 to analyze the page and identify form fields
            prompt = f"""
            Analyze this HTML and identify all the select dropdown fields.
            Also fill in the values for the input using the preferences.
            For each input you need to fill in the value of the input.
            
            For a select input the possible values are a boolean (true/false)
            
            Return a JSON object with ApplicationField objects.
            
            CRITICAL GUIDELINES:
            1. Find all the form inputs related to the job application.
            2. Use proper CSS selectors that target the ACTUAL form elements (input, select, textarea, etc.).
            3. DO NOT include iframe selectors like "iframe#grnhse_iframe" - only target the form elements themselves.
            4. If form elements are inside an iframe, use selectors that would work within that iframe context.
            5. Examples of good selectors: "input[name='name']", "select[id='city']", "input[type='email']"
            6. Examples of bad selectors: "iframe#grnhse_iframe", "iframe iframe input[name='name']"
            
            Preferences: {self.preferences}
            HTML of the page: {page_content}
            """
            
            response = client.responses.parse(
                model="gpt-4.1",
                input=[
                    {"role": "system", "content": "You are an expert at analyzing HTML and identifying form inputs. Return only valid JSON with CSS selectors."},
                    {"role": "user", "content": prompt}
                ],
                text_format=ApplicationFieldResponse
            )
            
            all_fields = response.output_parsed.fields
            
            all_fields = [field for field in all_fields if field.field_is_visible]
            
            # Filter out fields that are not visible
            context_str = "iframe" if frame else "page"
            
            # Parse the JSON response
            select_fields: List[ApplicationField] = [field for field in all_fields if field.field_type == FieldType.SELECT]
            
            context_str = "iframe" if frame else "page"
            print(f"✅ gpt-4.1 found {len(select_fields)} select fields in {context_str}")
            
            return select_fields
            
        except Exception as e:
            print(f"❌ Error in find_and_handle_all_select_fields: {e}")
            return []

    def find_upload_file_button(self, frame=None) -> ApplicationField:
        """
        Find upload file button in page or iframe context
        
        Args:
            frame: Optional iframe frame context. If None, uses main page.
        """
        try:
            import openai
            client = openai.OpenAI()
            
            # Get page content from frame or main page
            if frame:
                page_content = frame.content()
            else:
                page_content = self.page.content()
            
            # Use gpt-4.1 to analyze the page and identify form fields
            prompt = f"""
            Analyze this HTML and identify the upload file button.
            Also fill in the values for the input using the preferences.
            For each input you need to fill in the value of the input.
            
            For a upload input the possible values are a boolean (true/false)
            
            Return a JSON object with ApplicationField object.
            
            CRITICAL GUIDELINES:
            1. Find all the form inputs related to the job application.
            2. Use proper CSS selectors that target the ACTUAL form elements (input, select, textarea, etc.).
            3. DO NOT include iframe selectors like "iframe#grnhse_iframe" - only target the form elements themselves.
            4. If form elements are inside an iframe, use selectors that would work within that iframe context.
            5. Examples of good selectors: "input[name='name']", "select[id='city']", "input[type='email']"
            6. Examples of bad selectors: "iframe#grnhse_iframe", "iframe iframe input[name='name']"
            
            Preferences: {self.preferences}
            HTML of the page: {page_content}
            """
            
            response = client.responses.parse(
                model="gpt-4.1",
                input=[
                    {"role": "system", "content": "You are an expert at analyzing HTML and identifying form inputs. Return only valid JSON with CSS selectors."},
                    {"role": "user", "content": prompt}
                ],
                text_format=ApplicationField
            )
            
            application_field = response.output_parsed
            
            # Parse the JSON response
            return application_field
            
        except Exception as e:
            print(f"❌ Error in find_upload_file_button: {e}")
            return None

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

    def find_all_form_inputs(self, frame=None) -> Dict[str, List[ApplicationField]]:
        """
        Find all types of form inputs on the current page or in iframe using unified field detection
        
        Args:
            frame: Optional iframe frame context. If None, uses main page.
            
        Returns:
            Dict containing lists of different field types
        """
        try:
            # Use unified field detection functions
            text_input_fields = self.find_all_text_input_fields(frame)
            selectors = self.find_and_handle_all_select_fields(frame)
            radio_groups = self.find_all_radio_fields(frame)
            checkboxes = self.find_all_checkbox_fields(frame)
            
            # For upload buttons, we need to handle the single return value
            upload_button = self.find_upload_file_button(frame)
            
            total_fields = (len(text_input_fields) + len(selectors) + len(radio_groups) + 
                            len(checkboxes) + len([upload_button]))
            
            context_str = "iframe" if frame else "page"
            print(f"✅ Unified field detection found {total_fields} total form fields in {context_str}")
            print(f"  - {len(text_input_fields)} text input fields")
            print(text_input_fields)
            print(f"  - {len(selectors)} select dropdowns")
            print(selectors)
            print(f"  - {len(radio_groups)} radio button groups")
            print(radio_groups)
            print(f"  - {len(checkboxes)} checkboxes")
            print(checkboxes)
            print(f"  - {len([upload_button])} upload buttons")
            print(upload_button)
            
            return {
                'text_input_fields': text_input_fields,
                'selectors': selectors, 
                'radios': radio_groups,
                'checkboxes': checkboxes,
                'upload_button': upload_button,
                'total_fields': total_fields
            }
            
        except Exception as e:
            context_str = "iframe" if frame else "page"
            print(f"❌ Error finding form fields with unified detection in {context_str}: {e}")


# Example usage and testing
if __name__ == "__main__":
    # This would be integrated with the main HybridBrowserBot
    print("🧪 Application Filler module loaded")
    print("This module should be imported and used with HybridBrowserBot") 