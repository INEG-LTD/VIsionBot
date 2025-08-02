#!/usr/bin/env python3
"""
Application Filler

A comprehensive job application form filler that can handle various form types
and automatically fill them based on user preferences using gemini-2.5-pro analysis.
"""

# Removed incorrect import
import time
import json
from typing import List, Dict, Optional, Any, Tuple, Union
from playwright.sync_api import Page
import openai
from pydantic import BaseModel
from enum import Enum
import traceback
import os
import difflib
from google import genai

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
    field_value: Optional[bool] = None
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
                    frame = iframe_context['iframe_context']['frame']
                    text_fields, select_fields, radio_fields, checkbox_fields, upload_fields = self.find_all_form_inputs(frame)
                else:
                    # Use unified field detection functions for main page
                    text_fields, select_fields, radio_fields, checkbox_fields, upload_fields = self.find_all_form_inputs()
                
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
                success = self.fill_all_form_inputs(text_fields, select_fields, radio_fields, checkbox_fields, upload_fields, frame)
                if not success:
                    print("❌ Failed to fill form inputs")
                    return False
                
                # Step 4: Find a next button if there is one
                next_button = self.find_and_click_next_button()
                if not next_button:
                    print("❌ Failed to find or click next button")
                    print("🔍 Checking for submit button")
                else:
                    print("✅ Next button found and clicked")
                    state_result = self.check_form_submission_with_gpt(frame)
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
                        print("🎉 Application process completed!")
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
                               
                # Step 6: Take screenshot and analyze submission
                screenshot_path = f"form_submission_{self.current_iteration}.png"
                self.page.screenshot(path=screenshot_path, full_page=True)
                print(f"📸 Screenshot saved: {screenshot_path}")
                
                # Step 7: Analyze if form was submitted successfully
                submission_result = self.check_form_submission_with_gpt(frame)
                
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
        
    def fill_all_form_inputs(self, text_fields: List[TextApplicationField], select_fields: List[SelectApplicationField], radio_fields: List[RadioApplicationField], checkbox_fields: List[CheckboxApplicationField], upload_fields: List[UploadApplicationField], frame=None) -> bool: 
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
                    self.handle_file_upload(field, frame)
            
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
            
            # Check if this is a system or preference field that should be skipped
            if self._is_system_or_preference_field(select_field):
                context_str = "iframe" if frame else "page"
                print(f"⏭️ Skipping system/preference select field in {context_str}: {select_field.field_name}")
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
                    
                    result = self._find_and_click_option_with_gpt(frame.content(), self.preferences, frame)
                    if result:
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

    def _find_dropdown_field_with_gpt(self, page_html: str, original_selector: str, target_value: str, context: str) -> bool:
        """Use gemini-2.5-pro to find the dropdown field from HTML snapshot"""
        try:
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
            
            client = genai.Client(api_key="AIzaSyAU6PHwVlJJV5kogd4Es9hNf2Xy74fAOiA")
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction="You are an expert at analyzing HTML and finding dropdown fields. Return only valid JSON."
                )
            )
            
            result = response.text.strip()
            
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
                        print(f"✅ gemini-2.5-pro found dropdown field: {dropdown_data['selector']}")
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
            print(f"❌ Error in gemini-2.5-pro dropdown field detection: {e}")
            return None
    
    def _find_and_click_option_with_gpt(self, page_html: str, preferences: str, frame) -> Optional[Dict[str, Any]]:
        """Use gemini-2.5-pro to find and click the correct option from HTML snapshot"""
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
                ✅ **GOOD examples** (unique)  
                • `div[role="option"]:has-text("United Kingdom")`  
                • `#react-select-3-option-5`  
                • `div.css-a1b2c3-option[data-value="ca"]`  
                • `div[role="option"][data-value="fr"]`  

                ❌ **BAD examples** (non-unique or fragile)  
                • `div[role="option"]`                — matches every option  
                • `.css-123456`                       — class may apply to many nodes  
                • `div[role="option"]:first-child`   — position breaks if list changes  
                • `#react-select-3-option-`           — partial id, not unique

                ------------------------------------------------------------------------
                **Return format**  
                Exactly the selector string, nothing else.

                HTML of the open dropdown:
                {page_html}
                """
            
            client = genai.Client(api_key="AIzaSyAU6PHwVlJJV5kogd4Es9hNf2Xy74fAOiA")
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction="You are an expert at analyzing HTML and finding dropdown options. Return only the selector that will find the option element."
                )
            )
            result = response.text.strip()
            
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
            print(f"❌ Error in gemini-2.5-pro option detection: {e}")
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
    
    def _is_system_or_preference_field(self, select_field: SelectApplicationField) -> bool:
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
    # These are no longer needed since we use gemini-2.5-pro for intelligent dropdown handling
    
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
    
    def _safe_click_element(self, element) -> bool:
        """
        Safely click an element with multiple fallback strategies
        """
        try:
            if not element:
                return False
            
            # Strategy 1: Scroll into view and direct click
            try:
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
                print(f"⏭️ Skipping non-interactive checkbox in {context_str}: {checkbox.field_name}")
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

    def click_upload_button(self, upload_button: UploadApplicationField, frame=None) -> bool:
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

    def handle_file_upload(self, upload_button: UploadApplicationField, frame=None, times_tried=0) -> bool:
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

    def find_submit_button_with_gpt(self, frame: Dict[str, Any] = None) -> SubmitButtonApplicationField:
        """
        Use gemini-2.5-pro to contextually find submit buttons on the page or in iframe
        
        Args:
            iframe_context: Iframe context if searching within an iframe
            
        Returns:
            SubmitButtonApplicationField with submit button information and action
        """
        try:
            print("🤖 Using gemini-2.5-pro to find submit button contextually...")
            
            if frame:
                page_content = frame.content()
            else:
                page_content = self.page.content()
            
            client = genai.Client(api_key="AIzaSyAU6PHwVlJJV5kogd4Es9hNf2Xy74fAOiA")
            
            prompt = f"""
            
                1.	TASK
            Locate the single HTML element that submits or advances a job-application form and return it as an instantiation of SubmitButtonApplicationField.
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
            • Allowed element types: , , or any tag with role=“button”.
            • Text cues (case-insensitive): submit, continue, next, apply, send, save, finish, complete.
            • Attribute cues: type=“submit”; class/id containing a text cue; data-action*=“submit”; onclick that submits a form or advances the process.
            • Preference hierarchy:

                1.	Prefer elements that are descendants of the primary .
                2.	If multiple candidates exist within that form, choose the one nearest the end of the form’s content flow.
                3.	If ties remain, prefer elements with explicit type=“submit” or a text cue from the list.

                4.	FIELD VALUE GUIDANCE
            • field_selector   — stable, concise CSS selector (id if present, otherwise a specific path).
            • field_is_visible — True if element is not hidden (no hidden attribute, not aria-hidden=“true”, not style/display:none, not visibility:hidden); False if hidden; None if unknown.
            • field_in_form    — True if the element is inside a ; False otherwise; None if unknown.
            • field_required   — True only if the element explicitly has required or aria-required=“true”; otherwise False or None if unknown.
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
            Correct behaviour: choose the element that advances the flow (“Next”); if ambiguous, pick the one nearest the end of the form.
                6.	INPUT
            HTML source:
            {page_content}
            """
            
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction="You are an expert at analyzing HTML and identifying submit buttons for job application forms. Return only valid JSON.",
                    response_mime_type="application/json",
                    response_schema=SubmitButtonApplicationField
                )
            )
            
            result = response.parsed
            
            print(f"🤖 gemini-2.5-pro found submit button: {result}")
            
            return result
            
        except Exception as e:
            print(f"❌ Error in GPT submit button detection: {e}")
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
  
    def check_form_submission_with_gpt(self, frame: Dict[str, Any] = None) -> ApplicationStateResponse:
        """
        Use gemini-2.5-pro to analyze if the form was successfully submitted
        
        Args:
            frame: Frame to analyze
            
        Returns:
            ApplicationStateResponse with submission status and categorized fields
        """
        try:
            # For now, use page content analysis instead of image analysis
            # In a real implementation, you could use GPT-4 Vision API
            
            if frame:
                page_content = frame.content()
            else:
                page_content = self.page.content()
            
            client = genai.Client(api_key="AIzaSyAU6PHwVlJJV5kogd4Es9hNf2Xy74fAOiA")
            
            prompt = f"""
                Analyse the web page below and output a STRICTLY-minified JSON object that conforms to this schema:

                {{
                "submitted":            bool,  # form data accepted
                "completed":            bool,  # application fully finished
                "error_in_submission":  bool,  # any submission/validation/server error present
                "verification_required":bool,  # login/OTP/captcha or similar gate
                "more_forms":           bool   # additional steps or forms still visible
                }}

                Guidelines  
                - Success-phrases (“thank you”, “application submitted”, confirmation URLs) → submitted = true  
                - “Application complete” / final confirmation page → completed = true  
                - Error messages or HTTP errors → error_in_submission = true  
                - Requests for sign-in, OTP, captcha, identity check → verification_required = true  
                - “Next”, “Continue”, progress bars, or extra form elements → more_forms = true  
                - If completed is true then submitted must be true and more_forms must be false.

                === PAGE CONTENT ===
                {page_content}
                === END PAGE CONTENT ===

                Return the JSON only - no extra text, comments, or formatting.
                """
            
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction="You are an expert at analyzing job application pages. Return only valid JSON.",
                    response_mime_type="application/json",
                    response_schema=ApplicationStateResponse
                )
            )
            
            result: ApplicationStateResponse = response.parsed
            
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

    def find_all_text_input_fields(self, frame=None) -> List[TextApplicationField]:
        """
        Find all text input fields in page or iframe context
        
        Args:
            frame: Optional iframe frame context. If None, uses main page.
        """
        try:
            from google import genai
            client = genai.Client(api_key="AIzaSyAU6PHwVlJJV5kogd4Es9hNf2Xy74fAOiA")
            
            # Get page content from frame or main page
            if frame:
                page_content = frame.content()
            else:
                page_content = self.page.content()
            
            # Use gemini-2.5-pro to analyze the page and identify form fields
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
            7. VERY IMPORTANT - Only include INTERACTIVE and VISIBLE fields:
                - Do NOT include fields that are hidden by CSS (display: none, visibility: hidden, opacity: 0)
                - Do NOT include fields that are positioned off-screen or have zero dimensions
                - Do NOT include fields that are disabled or not enabled
                - Do NOT include fields from future form steps that aren't visible yet
                - Only include fields that a user can actually see and interact with on the current page
            8. For each field, set field_is_visible=true and field_in_form=true only if the field is actually visible and interactive
            
            Preferences: {self.preferences}
            HTML of the page: {page_content}
            """
            
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction="You are an expert at analyzing HTML and identifying form inputs. Return only valid JSON with CSS selectors.",
                    response_mime_type="application/json",
                    response_schema=TextApplicationFieldResponse
                )
            )
            
            all_fields = response.parsed.fields
            
            all_fields = [field for field in all_fields if field.field_is_visible and field.field_in_form]
            
            # Parse the JSON response
            text_input_fields: List[TextApplicationField] = [field for field in all_fields if field.field_type == FieldType.TEXT]
            
            context_str = "iframe" if frame else "page"
            print(f"✅ gemini-2.5-pro found {len(text_input_fields)} text input fields in {context_str}")
            
            return text_input_fields
            
        except Exception as e:
            print(f"❌ Error in find_all_text_input_fields: {e}")
            return []

    def find_all_radio_fields(self, frame=None) -> List[RadioApplicationField]:
        """
        Find all radio fields in page or iframe context
        
        Args:
            frame: Optional iframe frame context. If None, uses main page.
        """
        try:
            client = genai.Client(api_key="AIzaSyAU6PHwVlJJV5kogd4Es9hNf2Xy74fAOiA")
            
            # Get page content from frame or main page
            if frame:
                page_content = frame.content()
            else:
                page_content = self.page.content()
            
            # Use gemini-2.5-pro to analyze the page and identify form fields
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
            7. VERY IMPORTANT - Only include INTERACTIVE and VISIBLE fields:
                - Do NOT include fields that are hidden by CSS (display: none, visibility: hidden, opacity: 0)
                - Do NOT include fields that are positioned off-screen or have zero dimensions
                - Do NOT include fields that are disabled or not enabled
                - Do NOT include fields from future form steps that aren't visible yet
                - Only include fields that a user can actually see and interact with on the current page
            8. For each field, set field_is_visible=true and field_in_form=true only if the field is actually visible and interactive
            
            Preferences: {self.preferences}
            HTML of the page: {page_content}
            """
            
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction="You are an expert at analyzing HTML and identifying form inputs. Return only valid JSON with CSS selectors.",
                    response_mime_type="application/json",
                    response_schema=RadioApplicationFieldResponse
                )
            )
            
            all_fields = response.parsed.fields
            
            all_fields = [field for field in all_fields if field.field_is_visible and field.field_in_form]
            
            context_str = "iframe" if frame else "page"
            
            # Parse the JSON response
            radio_fields: List[RadioApplicationField] = [field for field in all_fields if field.field_type == FieldType.RADIO]
            
            print(f"✅ gemini-2.5-pro found {len(radio_fields)} radio fields in {context_str}")
            
            return radio_fields
            
        except Exception as e:
            print(f"❌ Error in find_all_radio_fields: {e}")
            return []

    def find_all_checkbox_fields(self, frame=None) -> List[CheckboxApplicationField]:
        """
        Find all checkbox fields in page or iframe context
        
        Args:
            frame: Optional iframe frame context. If None, uses main page.
        """
        try:
            client = genai.Client(api_key="AIzaSyAU6PHwVlJJV5kogd4Es9hNf2Xy74fAOiA")
            
            # Get page content from frame or main page
            if frame:
                page_content = frame.content()
            else:
                page_content = self.page.content()
            
            # Use gemini-2.5-pro to analyze the page and identify form fields
            prompt = f"""
            Analyze this HTML and identify all the checkbox fields.
            Also fill in the values for the input using the preferences.
            For each input you need to fill in the value of the input.
            
            For a checkbox input the possible field_value must be a boolean (true/false)
            
            Return a JSON object with ApplicationField objects.
            
            CRITICAL GUIDELINES:
            1. Find all the form inputs related to the job application.
            2. Use proper CSS selectors that target the ACTUAL form elements (input, select, textarea, etc.).
            3. DO NOT include iframe selectors like "iframe#grnhse_iframe" - only target the form elements themselves.
            4. If form elements are inside an iframe, use selectors that would work within that iframe context.
            5. Examples of good selectors: "input[name='name']", "select[id='city']", "input[type='email']"
            6. Examples of bad selectors: "iframe#grnhse_iframe", "iframe iframe input[name='name']"
            7. VERY IMPORTANT - Only include INTERACTIVE and VISIBLE fields:
                - Do NOT include fields that are hidden by CSS (display: none, visibility: hidden, opacity: 0)
                - Do NOT include fields that are positioned off-screen or have zero dimensions
                - Do NOT include fields that are disabled or not enabled
                - Do NOT include fields from future form steps that aren't visible yet
                - Only include fields that a user can actually see and interact with on the current page
            8. For each field, set field_is_visible=true and field_in_form=true only if the field is actually visible and interactive
            
            Preferences: {self.preferences}
            HTML of the page: {page_content}
            """
            
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction="You are an expert at analyzing HTML and identifying form inputs. Return only valid JSON with CSS selectors.",
                    response_mime_type="application/json",
                    response_schema=CheckboxApplicationFieldResponse
                )
            )
            
            all_fields = response.parsed.fields
            
            all_fields = [field for field in all_fields if field.field_is_visible and field.field_in_form]
            
            context_str = "iframe" if frame else "page"
            # Parse the JSON response
            checkbox_fields: List[CheckboxApplicationField] = [field for field in all_fields if field.field_type == FieldType.CHECKBOX]
            
            print(f"✅ gemini-2.5-pro found {len(checkbox_fields)} checkbox fields in {context_str}")
            
            return checkbox_fields
            
        except Exception as e:
            print(f"❌ Error in find_all_checkbox_fields: {e}")
            return []

    def find_all_select_fields(self, frame=None) -> List[SelectApplicationField]:
        """
        Find all select fields in page or iframe context
        
        Args:
            frame: Optional iframe frame context. If None, uses main page.
        """
        try:
            client = genai.Client(api_key="AIzaSyAU6PHwVlJJV5kogd4Es9hNf2Xy74fAOiA")
            
            # Get page content from frame or main page
            if frame:
                page_content = frame.content()
            else:
                page_content = self.page.content()
            
            # Use gemini-2.5-pro to analyze the page and identify form fields
            prompt = f"""
                You are an expert HTML form analyzer. Your task is to extract ONLY SELECT dropdown input fields from the given HTML of a job application form and return them in strict JSON format.
                Your output should be suitable for python playwright and will be used like this:
                    if select_field.field_selector.startswith('#'):
                        if frame:
                            element = frame.locator(f'id={{select_field.field_selector[1:]}}').first
                        else:
                            element = self.page.locator(f'id={{select_field.field_selector[1:]}}').first
                    else:
                        if frame:
                            element = frame.locator(select_field.field_selector)
                        else:
                            element = self.page.locator(select_field.field_selector)
                
                Follow these instructions precisely:

                1 GOAL
                - Detect all dropdown fields that let a user choose one or multiple options.
                - Detect both:
                - Native <select> dropdowns with <option> tags.
                - React Select dropdowns rendered as a clickable <div> (usually with class*="select__control") that opens a dropdown menu.

                2 SELECTION RULES
                - Include only dropdown fields that are:
                - Inside the job application form.
                - Visible and interactive for the user (not hidden, disabled, or off-screen).
                - Currently displayed on the page (ignore future-step fields not yet visible).
                - For native selects: Target the <select> element directly using a unique, minimal CSS selector.
                Examples of GOOD field_selector:
                    ✅ "select[name='country']"
                    ✅ "select#job_type"
                    ✅ "form#application-form select[name='department']"
                - For React Select dropdowns: Target the main clickable container of the component.
                Examples of GOOD field_selector:
                    ✅ "div.select__control"
                    ✅ "div[class*='select__control']"
                    ✅ "form#job-application div.select__control"
                - BAD field_selector examples (never include these):
                    ❌ "iframe#grnhse_iframe select[name='country']"                            (iframe references are not allowed)
                    ❌ "body iframe iframe div.select__control"                                 (cross-document selectors)
                    ❌ "div.select__control[aria-labelledby="question_32214839002-label"]"      (aria-labelledby is not a select dropdown)
                    ❌ "html body div"                                                          (too generic, does not specifically target the field)
                    ❌ "div.select__menu"                                                       (menu list, not the interactive control element)
                    ❌ "span" or "label"                                                        (not the actual interactive field)

                3 OUTPUT FORMAT
                - field_selector → Minimal, stable CSS selector directly targeting the dropdown control (never include iframes or overly generic selectors).
                - field_name → Name attribute or visible label for the field.
                - field_type → Always "select".
                - field_is_visible → true only if visible and interactive.
                - field_in_form → true only if part of the job application form.
                - field_value → Leave empty.
                - field_options → List visible options if available, otherwise [].

                4 CRITICAL CONSTRAINTS
                - Accuracy is mandatory: Do not guess fields or selectors that don't exist.
                - No hidden fields: Skip elements with display:none, visibility:hidden, zero size, or off-screen positioning.
                - No iframes: Never include "iframe#..." or cross-document selectors.
                - No other input types: Only return select dropdown fields, ignore text, checkbox, radio, file upload, etc.

                Input Data:
                - Preferences: {self.preferences}
                - HTML of the page: {page_content}

                Expected behavior:
                - A clean, strictly valid JSON array with only select dropdowns, each fully described according to the schema above.
                - No extra commentary or explanation in the output—just valid JSON.
                """
            
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction="You are an expert at analyzing HTML and identifying form inputs. Return only valid JSON with CSS selectors.",
                    response_mime_type="application/json",
                    response_schema=SelectApplicationFieldResponse
                )
            )
            
            all_fields = response.parsed.fields
            
            all_fields = [field for field in all_fields if field.field_is_visible and field.field_in_form]
            
            # Filter out fields that are not visible
            context_str = "iframe" if frame else "page"
            
            # Parse the JSON response
            select_fields: List[SelectApplicationField] = [field for field in all_fields if field.field_type == FieldType.SELECT]
            
            context_str = "iframe" if frame else "page"
            print(f"✅ gemini-2.5-pro found {len(select_fields)} select fields in {context_str}")
            
            return select_fields
            
        except Exception as e:
            print(f"❌ Error in find_and_handle_all_select_fields: {e}")
            traceback.print_exc()
            return []

    def find_upload_file_button(self, frame=None) -> UploadApplicationField:
        """
        Find upload file button in page or iframe context
        
        Args:
            frame: Optional iframe frame context. If None, uses main page.
        """
        try:
            client = genai.Client(api_key="AIzaSyAU6PHwVlJJV5kogd4Es9hNf2Xy74fAOiA")
            
            # Get page content from frame or main page
            if frame:
                page_content = frame.content()
            else:
                page_content = self.page.content()
            
            # Use gemini-2.5-pro to analyze the page and identify form fields
            prompt = f"""
                ##########  CONTEXT  ##########
                You are a senior front-end engineer.  
                Your task: scan the supplied HTML and return **exactly one** CSS selector that matches the user-visible control that lets an applicant upload a file (resume, CV, cover letter, etc.).

                The upload control may be a <button>, <a>, or <div> acting as a button.  
                **Never** return an <input> selector.

                ##########  CONSTRAINTS  ##########
                1. Target the real form element — not wrappers or generic containers.  
                2. Ignore every <input>, even if it is type="file".  
                3. **Do not** prepend iframe selectors (e.g. `iframe#grnhse_iframe …`).  
                • If the control lives inside an iframe, write the selector **as if you are already inside** that iframe's DOM.  
                4. Exclude controls that are hidden (display:none, aria-hidden="true", etc.) or that belong to steps not yet visible.  
                5. Use robust attribute / class patterns; avoid brittle nth-child or positional selectors unless unavoidable.  

                ##########  GOOD FIELD SELECTOR EXAMPLES  ##########
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

                ##########  BAD FIELD SELECTOR EXAMPLES  ##########
                input[type='file']                          # input, not button/link/div  
                iframe#grnhse_iframe button[type='file']    # crosses iframe boundary  
                div                                         # hopelessly generic  
                body button                                 # far too broad  
                button:nth-child(3)                         # brittle index-based  
                div[aria-labelledby='upload-label-resume'] .button-container button  # wrapper, not actual control  
                .modal[style*='display:none'] button        # hidden element  
                form:first-of-type button[type='submit']    # submit button, not upload  
                a[href='#']                                 # generic anchor  

                ##########  DATA  ##########
                User-specific preferences:
                {self.preferences}

                HTML to analyse
                {page_content}
                """
            
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction="You are an expert at analyzing HTML and identifying form inputs. Return only valid JSON with CSS selectors.",
                    response_mime_type="application/json",
                    response_schema=UploadApplicationField
                )
            )
            
            upload_field = response.parsed
            
            # Parse the JSON response
            return upload_field
            
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

    def find_all_form_inputs(self, frame=None) -> Tuple[List[TextApplicationField], List[SelectApplicationField], List[RadioApplicationField], List[CheckboxApplicationField], List[UploadApplicationField]]:
        """
        Find all types of form inputs on the current page or in iframe using unified field detection
        
        Args:
            frame: Optional iframe frame context. If None, uses main page.
            
        Returns:
            Tuple with categorized fields
        """
        try:
            # Use unified field detection functions
            text_input_fields = self.find_all_text_input_fields(frame)
            selectors = self.find_all_select_fields(frame)
            radio_groups = self.find_all_radio_fields(frame)
            checkboxes = self.find_all_checkbox_fields(frame)
            
            # For upload buttons, we need to handle the single return value
            upload_button = self.find_upload_file_button(frame)
            upload_fields = [upload_button] if upload_button is not None else []
            
            total_fields = (len(text_input_fields) + len(selectors) + len(radio_groups) + 
                            len(checkboxes) + len(upload_fields))
            
            context_str = "iframe" if frame else "page"
            print(f"✅ Unified field detection found {total_fields} total form fields in {context_str}")
            print(f"  - {len(text_input_fields)} text input fields")
            print(f"  - {len(selectors)} select dropdowns")
            print(f"  - {len(radio_groups)} radio button groups")
            print(f"  - {len(checkboxes)} checkboxes")
            print(f"  - {len(upload_fields)} upload buttons")
            
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
            print(f"❌ Error finding form fields with unified detection in {context_str}: {e}")
            return [], [], [], [], []

    def _find_alternative_select_value(self, possible_values: List[str], select_field: SelectApplicationField, preferences: Dict[str, Any]) -> Optional[str]:
        """
        Use gemini-2.5-pro to find an alternative value from the available options when the original value is not found
        
        Args:
            possible_values: List of available values in the select field
            select_field: The select field that needs an alternative value
            preferences: User preferences for filling forms
            
        Returns:
            Optional[str]: Alternative value if found, None otherwise
        """
        try:
            client = genai.Client(api_key="AIzaSyAU6PHwVlJJV5kogd4Es9hNf2Xy74fAOiA")
            
            field_name = select_field.field_name or ""
            page_content = self.page.content()
            print(f"🤖 Using gemini-2.5-pro to find alternative value for field: {field_name}")
            print(f"   Available values: {possible_values}")
            
            prompt = f"""
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
            
            response = client.models.generate_content(
                model="gemini-2.5-pro",
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    system_instruction="You are an expert at analyzing form fields and selecting appropriate values. Return only the selected value as a string."
                )
            )
            
            selected_value = response.text.strip()
            
            if not possible_values:
                return selected_value
            
            # Verify the selected value is actually in the available options
            if selected_value in possible_values:
                print(f"✅ gemini-2.5-pro selected: {selected_value}")
                return selected_value
            else:
                # Try case-insensitive match
                for value in possible_values:
                    if value.lower() == selected_value.lower():
                        print(f"✅ gemini-2.5-pro selected (case-insensitive): {value}")
                        return value
                
                # If still no match, use the first valid option
                for value in possible_values:
                    if value and value.strip() and value.lower() not in ['select', 'choose', 'please select', '--', '']:
                        print(f"⚠️ gemini-2.5-pro selection '{selected_value}' not found, using fallback: {value}")
                        return value
            
            print(f"❌ No suitable alternative value found for {field_name}")
            return None
            
        except Exception as e:
            print(f"❌ Error in gemini-2.5-pro alternative value selection: {e}")
            # Fallback to first valid option
            for value in possible_values:
                if value and value.strip() and value.lower() not in ['select', 'choose', 'please select', '--', '']:
                    print(f"🔄 Using fallback value: {value}")
                    return value
            return None


# Example usage and testing
if __name__ == "__main__":
    # This would be integrated with the main HybridBrowserBot
    print("🧪 Application Filler module loaded")
    print("This module should be imported and used with HybridBrowserBot") 