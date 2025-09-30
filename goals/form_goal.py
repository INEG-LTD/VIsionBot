"""
Form Fill Goal - Monitors and validates form completion.
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional

from playwright.sync_api import Page

from .base import BaseGoal, GoalResult, GoalStatus, GoalContext, EvaluationTiming
from utils.vision_resolver import rank_elements_against_instruction


class FormFillGoal(BaseGoal):
    """
    Goal that monitors form filling and validates field completion.
    
    This goal can evaluate in two modes:
    1. After field input - validates each field as it's filled
    2. After submit attempt - validates entire form and handles errors
    """
    
    EVALUATION_TIMING = EvaluationTiming.BOTH
    
    def __init__(
        self, 
        description: str,
        required_fields: Optional[List[str]] = None,
        trigger_on_submit: bool = True,
        trigger_on_field_input: bool = True,
        **kwargs
    ):
        super().__init__(description, **kwargs)
        self.required_fields = required_fields or []  # Specific fields to monitor
        self.trigger_on_submit = trigger_on_submit
        self.trigger_on_field_input = trigger_on_field_input
        
        # State tracking
        self.detected_fields: List[Dict[str, Any]] = []
        self.field_values: Dict[str, str] = {}
        self.submit_button_available = False
        self.next_button_available = False
        self.form_errors: List[str] = []
        self.last_field_check = 0
    
    def evaluate(self, context: GoalContext) -> GoalResult:
        """Evaluate form completion status"""
        
        page = context.page_reference
        if not page:
            return GoalResult(
                status=GoalStatus.UNKNOWN,
                confidence=0.0,
                reasoning="No page reference available for form evaluation"
            )
        
        # Detect current form elements
        self.detected_fields = self.detect_elements_on_page(page, self.description)
        
        # Filter to get relevant fields for this goal
        relevant_fields = self._filter_relevant_fields(context, self.detected_fields)
        
        # Update field values for relevant fields only
        self._update_field_values(page, relevant_fields)
        
        # Detect form errors
        form_errors = self._detect_form_errors(page)
        
        # Detect submit/next buttons
        submit_info = self._detect_submit_and_next_buttons(page)
        
        # Count relevant form fields (input, select, textarea only)
        relevant_form_fields = [f for f in relevant_fields if f['tagName'] in ['input', 'select', 'textarea']]
        filled_relevant_fields = [f for f in relevant_form_fields if f.get('isFilled')]
        
        print("🔍 Form Analysis:")
        print(f"   📝 Total fields detected: {len(self.detected_fields)}")
        print(f"   🎯 Relevant fields for goal: {len(relevant_form_fields)}")
        print(f"   ✅ Relevant fields filled: {len(filled_relevant_fields)}")
        print(f"   🚫 Form errors: {len(form_errors)}")
        print(f"   📤 Submit available: {submit_info['submit_available']}")
        print(f"   ➡️  Next available: {submit_info['next_available']}")
        
        # Show field details for debugging
        if relevant_form_fields:
            print("   📋 Relevant field details:")
            for field in relevant_form_fields:
                field_name = field.get('name', field.get('id', field.get('placeholder', 'unnamed')))
                field_type = field.get('type', 'text')
                is_filled = field.get('isFilled', False)
                status = "✅" if is_filled else "⏳"
                print(f"      {status} {field_name} ({field_type})")
        
        # Determine evaluation based on interaction type and trigger settings
        last_interaction = context.all_interactions[-1] if context.all_interactions else None
        
        if last_interaction and last_interaction.interaction_type.value == "click":
            # After click - check if it was submit/next button
            if self.trigger_on_submit:
                return self._evaluate_after_click(context, form_errors, submit_info, relevant_fields)
            else:
                # Submit trigger disabled, return pending
                return GoalResult(
                    status=GoalStatus.PENDING,
                    confidence=0.5,
                    reasoning="Click interaction occurred but submit trigger is disabled for this goal"
                )
        elif last_interaction and last_interaction.interaction_type.value == "type":
            # After typing - check field completion
            if self.trigger_on_field_input:
                return self._evaluate_after_field_input(context, form_errors, submit_info, relevant_fields)
            else:
                # Field input trigger disabled, return pending
                return GoalResult(
                    status=GoalStatus.PENDING,
                    confidence=0.5,
                    reasoning="Field input occurred but field input trigger is disabled for this goal"
                )
        else:
            # General evaluation - always allowed
            return self._evaluate_general_state(context, form_errors, submit_info, relevant_fields)
    
    def _update_field_values(self, page: Page, relevant_fields: List[Dict[str, Any]]) -> None:
        """Update the current values of relevant fields only"""
        self.field_values = {}
        
        for field in relevant_fields:
            if field['tagName'] in ['input', 'select', 'textarea']:
                coords = field['coordinates']
                value = self.get_field_value_at_coordinates(page, coords['x'], coords['y'])
                self.field_values[field.get('name', f"field_{field['index']}")] = value
                field['currentValue'] = value
                field['isFilled'] = bool(value and value.strip())
                field['isRelevant'] = True  # Mark as relevant to this goal
    
    def _filter_relevant_fields(self, context: GoalContext, all_fields: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter fields to only include those relevant to the goal description using AI"""
        if not all_fields:
            return []

        # Try deterministic vision-based matching first (fast path)
        vision_fields = self._vision_filter_relevant_fields(context, all_fields)
        if vision_fields:
            return vision_fields
        
        # Use AI to determine field relevance
        relevant_fields = self._ai_filter_relevant_fields(all_fields)
        
        print(f"   🎯 AI filtered to {len(relevant_fields)} relevant fields out of {len(all_fields)} total")
        return relevant_fields

    def _vision_filter_relevant_fields(self, context: GoalContext, all_fields: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Use deterministic vision heuristics to pick relevant form fields when possible."""
        page_w = context.current_state.page_width if context and context.current_state else None
        page_h = context.current_state.page_height if context and context.current_state else None

        if self.required_fields:
            selected_indices: List[int] = []
            for requirement in self.required_fields:
                ranked = rank_elements_against_instruction(
                    requirement,
                    all_fields,
                    page_w=page_w,
                    page_h=page_h,
                    mode="field",
                    interpretation_mode="semantic",
                )
                if not ranked:
                    continue
                idx, score, diag = ranked[0]
                if score < 5:
                    continue
                if idx not in selected_indices:
                    selected_indices.append(idx)
                field = all_fields[idx]
                field_name = field.get('name') or field.get('id') or field.get('placeholder') or field.get('textContent') or 'unnamed'
                print(f"   🎯 Vision matched '{requirement}' -> field index {idx} ({field_name}) [score={score:.1f}]")
            if selected_indices:
                return [all_fields[i] for i in selected_indices]

        # If no explicit requirements, pick high-scoring matches from the goal description itself
        ranked = rank_elements_against_instruction(
            self.description,
            all_fields,
            page_w=page_w,
            page_h=page_h,
            mode="field",
            interpretation_mode="semantic",
        )
        strong_candidates = [item for item in ranked if item[1] >= 6][: min(3, len(all_fields))]
        if strong_candidates:
            indices = [idx for idx, _, _ in strong_candidates]
            for idx, score, _ in strong_candidates:
                field = all_fields[idx]
                field_name = field.get('name') or field.get('id') or field.get('placeholder') or field.get('textContent') or 'unnamed'
                print(f"   🎯 Vision highlighted field index {idx} ({field_name}) from goal description [score={score:.1f}]")
            return [all_fields[i] for i in indices]

        return []

    def _ai_filter_relevant_fields(self, all_fields: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Use AI to determine which fields are relevant to the goal"""
        try:
            from ai_utils import generate_model
            from pydantic import BaseModel, Field
            
            # Create a simple model for AI response
            class FieldRelevance(BaseModel):
                relevant_field_indices: List[int] = Field(description="List of field indices that are relevant to the goal")
                reasoning: str = Field(description="Brief explanation of why these fields were selected")
            
            # Prepare field information for AI
            field_descriptions = []
            for i, field in enumerate(all_fields):
                field_info = {
                    "index": i,
                    "tagName": field.get('tagName', ''),
                    "type": field.get('type', ''),
                    "name": field.get('name', ''),
                    "id": field.get('id', ''),
                    "placeholder": field.get('placeholder', ''),
                    "ariaLabel": field.get('ariaLabel', ''),
                    "className": field.get('className', ''),
                    "textContent": field.get('textContent', ''),
                    "value": field.get('value', ''),
                    "required": field.get('required', False)
                }
                field_descriptions.append(field_info)
            
            system_prompt = f"""
            You are an expert at analyzing web forms and determining which fields are relevant to specific user goals.
            
            USER GOAL: "{self.description}"
            
            AVAILABLE FIELDS: {field_descriptions}
            
            Instructions:
            1. Analyze the user goal to understand what they want to accomplish
            2. Identify which form fields are directly relevant to achieving this goal
            3. Consider field names, types, placeholders, labels, and context
            4. Be precise - only include fields that are directly needed for the goal
            5. If the goal is general (like "fill the form"), include all fields
            6. If the goal is specific (like "set the time"), only include relevant fields
            
            Return the indices of relevant fields and explain your reasoning.
            """
            
            result = generate_model(
                prompt="Analyze the form fields and determine which ones are relevant to the user's goal.",
                model_object_type=FieldRelevance,
                reasoning_level="medium",
                system_prompt=system_prompt,
                model="gpt-5-nano"
            )
            
            if result and result.relevant_field_indices:
                relevant_fields = []
                for index in result.relevant_field_indices:
                    if 0 <= index < len(all_fields):
                        relevant_fields.append(all_fields[index])
                        print(f"   ✅ AI selected field {index}: {all_fields[index].get('name', all_fields[index].get('id', all_fields[index].get('placeholder', 'unnamed')))}")
                
                print(f"   🤖 AI reasoning: {result.reasoning}")
                return relevant_fields
            else:
                print("   ⚠️ AI couldn't determine field relevance, using all fields as fallback")
                return all_fields
                
        except Exception as e:
            print(f"   ⚠️ AI field filtering failed: {e}, using all fields as fallback")
            return all_fields
    
    def _detect_form_errors(self, page: Page) -> List[str]:
        """Detect form validation errors on the page"""
        try:
            js_code = """
            (function() {
                const errors = [];
                
                // Common error selectors
                const errorSelectors = [
                    '.error', '.field-error', '.form-error', '.validation-error',
                    '.alert-danger', '.text-danger', '.is-invalid', '.has-error',
                    '[role="alert"]', '.error-message', '.field-validation-error'
                ];
                
                errorSelectors.forEach(selector => {
                    const errorElements = document.querySelectorAll(selector);
                    errorElements.forEach(element => {
                        const text = element.textContent?.trim();
                        if (text && text.length > 0 && 
                            getComputedStyle(element).display !== 'none' &&
                            getComputedStyle(element).visibility !== 'hidden') {
                            errors.push({
                                text: text,
                                selector: selector,
                                visible: true
                            });
                        }
                    });
                });
                
                // Also check for invalid input states
                const invalidInputs = document.querySelectorAll('input:invalid, select:invalid, textarea:invalid');
                invalidInputs.forEach(input => {
                    const label = document.querySelector(`label[for="${input.id}"]`);
                    const fieldName = label?.textContent || input.name || input.placeholder || 'Unknown field';
                    errors.push({
                        text: `${fieldName} is invalid`,
                        selector: 'input:invalid',
                        visible: true,
                        fieldName: fieldName
                    });
                });
                
                return errors;
            })();
            """
            
            errors = page.evaluate(js_code)
            self.form_errors = [error['text'] for error in (errors or [])]
            return self.form_errors
            
        except Exception as e:
            print(f"⚠️ Error detecting form errors: {e}")
            return []
    
    def _detect_submit_and_next_buttons(self, page: Page) -> Dict[str, Any]:
        """Detect submit and next buttons and their availability"""
        try:
            js_code = """
            (function() {
                let submitButton = null;
                let nextButton = null;
                
                // Check all buttons
                const allButtons = document.querySelectorAll('button, input[type="submit"], [role="button"]');
                allButtons.forEach(button => {
                    const text = button.textContent?.toLowerCase() || '';
                    const className = button.className.toLowerCase();
                    const type = button.type || '';
                    
                    // Check if it's a submit button
                    if (type === 'submit' || 
                        text.includes('submit') || 
                        text.includes('send') ||
                        className.includes('submit')) {
                        
                        const rect = button.getBoundingClientRect();
                        if (rect.width > 0 && rect.height > 0) {
                            submitButton = {
                                element: button,
                                text: button.textContent?.trim() || '',
                                disabled: button.disabled,
                                visible: getComputedStyle(button).display !== 'none',
                                coordinates: {
                                    x: Math.round(rect.left + rect.width / 2),
                                    y: Math.round(rect.top + rect.height / 2)
                                }
                            };
                        }
                    }
                    
                    // Check if it's a next button
                    if (text.includes('next') || 
                        text.includes('continue') ||
                        className.includes('next')) {
                        
                        const rect = button.getBoundingClientRect();
                        if (rect.width > 0 && rect.height > 0) {
                            nextButton = {
                                element: button,
                                text: button.textContent?.trim() || '',
                                disabled: button.disabled,
                                visible: getComputedStyle(button).display !== 'none',
                                coordinates: {
                                    x: Math.round(rect.left + rect.width / 2),
                                    y: Math.round(rect.top + rect.height / 2)
                                }
                            };
                        }
                    }
                });
                
                return {
                    submitButton: submitButton,
                    nextButton: nextButton,
                    submit_available: submitButton && !submitButton.disabled && submitButton.visible,
                    next_available: nextButton && !nextButton.disabled && nextButton.visible
                };
            })();
            """
            
            result = page.evaluate(js_code)
            return result or {
                'submit_available': False,
                'next_available': False,
                'submitButton': None,
                'nextButton': None
            }
            
        except Exception as e:
            print(f"⚠️ Error detecting submit/next buttons: {e}")
            return {
                'submit_available': False,
                'next_available': False,
                'submitButton': None,
                'nextButton': None
            }
    
    def _evaluate_after_click(self, context: GoalContext, form_errors: List[str], submit_info: Dict[str, Any], relevant_fields: List[Dict[str, Any]]) -> GoalResult:
        """Evaluate form state after a click interaction"""
        
        # First, check for success indicators regardless of submit button detection
        # This handles cases where form submission navigated to a new page
        if self._detect_form_success(context.page_reference):
            print("   🎉 Form submission success detected!")
            return GoalResult(
                status=GoalStatus.ACHIEVED,
                confidence=0.9,
                reasoning="Form submitted successfully - detected success indicators on current page",
                evidence={"form_errors": form_errors, "success_detected": True}
            )
        
        # Check if the click was on a submit button (for cases where we're still on the form page)
        last_interaction = context.all_interactions[-1]
        click_coords = last_interaction.coordinates
        
        # Check if submit button was clicked
        was_submit_clicked = False
        if click_coords and submit_info.get('submitButton'):
            was_submit_clicked = self._was_submit_clicked(click_coords, submit_info)
        
        # If submit button was clicked, check for success indicators
        if was_submit_clicked:
            print("   📤 Submit button was clicked, checking for success...")
            
            # Check for form errors after submit
            if form_errors:
                # Request retry if we haven't exceeded max retries and there are fixable errors
                if self.can_retry() and len(form_errors) <= 3:  # Only retry for a few errors
                    retry_reason = f"Form has validation errors: {'; '.join(form_errors[:3])}"
                    if self.request_retry(retry_reason):
                        return GoalResult(
                            status=GoalStatus.PENDING,
                            confidence=0.7,
                            reasoning=f"Form submitted but has validation errors, requesting retry: {'; '.join(form_errors[:3])}",
                            evidence={"form_errors": form_errors, "retry_requested": True, "retry_count": self.retry_count},
                            next_actions=["Retry plan generation to fix validation errors"]
                        )
                
                return GoalResult(
                    status=GoalStatus.PENDING,
                    confidence=0.7,
                    reasoning=f"Form submitted but has validation errors: {'; '.join(form_errors[:3])}",
                    evidence={"form_errors": form_errors},
                    next_actions=["Fix validation errors and resubmit"]
                )
            
            # No clear success or error - might be processing
            return GoalResult(
                status=GoalStatus.PENDING,
                confidence=0.5,
                reasoning="Form submitted, waiting for response or checking for success indicators",
                evidence={"form_errors": form_errors}
            )
        
        # If next button was clicked, continue monitoring
        elif submit_info['next_available'] and self._was_next_clicked(click_coords, submit_info):
            print("   ➡️ Next button was clicked, continuing form monitoring...")
            
            unfilled_fields = [f for f in relevant_fields if not f.get('isFilled') and f['tagName'] in ['input', 'select', 'textarea']]
            
            return GoalResult(
                status=GoalStatus.PENDING,
                confidence=0.6,
                reasoning=f"Moved to next form step, {len(unfilled_fields)} fields still need filling",
                evidence={"unfilled_fields": len(unfilled_fields), "form_errors": form_errors}
            )
        
        # Other click - continue monitoring
        return GoalResult(
            status=GoalStatus.PENDING,
            confidence=0.4,
            reasoning="Click interaction completed, continuing form monitoring",
            evidence={"form_errors": form_errors}
        )
    
    def _evaluate_after_field_input(self, context: GoalContext, form_errors: List[str], submit_info: Dict[str, Any], relevant_fields: List[Dict[str, Any]]) -> GoalResult:
        """Evaluate form state after field input"""
        
        # Count filled vs unfilled relevant fields
        total_fields = len([f for f in relevant_fields if f['tagName'] in ['input', 'select', 'textarea']])
        filled_fields = len([f for f in relevant_fields if f.get('isFilled')])
        
        print(f"   📊 Field Progress: {filled_fields}/{total_fields} fields filled")
        
        # If all relevant fields are filled, mark as achieved regardless of form errors
        if filled_fields == total_fields and total_fields > 0:
            return GoalResult(
                status=GoalStatus.ACHIEVED,
                confidence=0.9,
                reasoning=f"All {total_fields} relevant fields are filled - goal completed",
                evidence={
                    "filled_fields": filled_fields,
                    "total_fields": total_fields,
                    "submit_available": submit_info['submit_available'],
                    "form_errors": len(form_errors)
                },
                next_actions=["Goal completed - relevant fields filled"] if not form_errors else ["Goal completed - relevant fields filled (form has validation errors on other fields)"]
            )
        
        # Still filling fields
        elif filled_fields < total_fields:
            unfilled_count = total_fields - filled_fields
            return GoalResult(
                status=GoalStatus.PENDING,
                confidence=0.5,
                reasoning=f"Form filling in progress: {filled_fields}/{total_fields} fields completed, {unfilled_count} remaining",
                evidence={
                    "filled_fields": filled_fields,
                    "total_fields": total_fields,
                    "unfilled_count": unfilled_count
                }
            )
        
        # No fields detected
        else:
            return GoalResult(
                status=GoalStatus.UNKNOWN,
                confidence=0.2,
                reasoning="No form fields detected on current page",
                evidence={"total_fields": total_fields}
            )
    
    def _evaluate_general_state(self, context: GoalContext, form_errors: List[str], submit_info: Dict[str, Any], relevant_fields: List[Dict[str, Any]]) -> GoalResult:
        """General form state evaluation"""
        
        total_fields = len([f for f in relevant_fields if f['tagName'] in ['input', 'select', 'textarea']])
        filled_fields = len([f for f in relevant_fields if f.get('isFilled')])
        
        # Check overall completion - prioritize relevant field completion
        if total_fields > 0:
            completion_ratio = filled_fields / total_fields
            
            # If all relevant fields are filled, mark as achieved regardless of form errors
            if completion_ratio >= 1.0:
                return GoalResult(
                    status=GoalStatus.ACHIEVED,
                    confidence=0.9,
                    reasoning=f"All {total_fields} relevant fields are filled - goal completed",
                    evidence={
                        "filled_fields": filled_fields,
                        "total_fields": total_fields,
                        "form_errors": len(form_errors)
                    },
                    next_actions=["Goal completed - relevant fields filled"] if not form_errors else ["Goal completed - relevant fields filled (form has validation errors on other fields)"]
                )
            else:
                # If there are form errors but not all relevant fields are filled, mention both
                if form_errors:
                    return GoalResult(
                        status=GoalStatus.PENDING,
                        confidence=0.5,
                        reasoning=f"Form filling in progress: {filled_fields}/{total_fields} relevant fields completed. Form has validation errors on other fields.",
                        evidence={"form_errors": form_errors, "filled_fields": filled_fields, "total_fields": total_fields},
                        next_actions=["Continue filling relevant fields"]
                    )
                else:
                    return GoalResult(
                        status=GoalStatus.PENDING,
                        confidence=0.4,
                        reasoning=f"Form filling in progress: {filled_fields}/{total_fields} relevant fields completed"
                    )
        
        return GoalResult(
            status=GoalStatus.UNKNOWN,
            confidence=0.2,
            reasoning="No clear form structure detected"
        )
    
    def _was_submit_clicked(self, click_coords: Optional[tuple], submit_info: Dict[str, Any]) -> bool:
        """Check if the last click was on the submit button"""
        if not click_coords or not submit_info.get('submitButton'):
            return False
        
        submit_coords = submit_info['submitButton']['coordinates']
        click_x, click_y = click_coords
        submit_x, submit_y = submit_coords['x'], submit_coords['y']
        
        # Check if click was within submit button area (allow some tolerance)
        distance = ((click_x - submit_x) ** 2 + (click_y - submit_y) ** 2) ** 0.5
        return distance < 50  # Within 50 pixels
    
    def _was_next_clicked(self, click_coords: Optional[tuple], submit_info: Dict[str, Any]) -> bool:
        """Check if the last click was on the next button"""
        if not click_coords or not submit_info.get('nextButton'):
            return False
        
        next_coords = submit_info['nextButton']['coordinates']
        click_x, click_y = click_coords
        next_x, next_y = next_coords['x'], next_coords['y']
        
        # Check if click was within next button area
        distance = ((click_x - next_x) ** 2 + (click_y - next_y) ** 2) ** 0.5
        return distance < 50  # Within 50 pixels
    
    def _detect_form_success(self, page: Page) -> bool:
        """Detect if form submission was successful"""
        try:
            js_code = """
            (function() {
                // Look for success indicators
                const successSelectors = [
                    '.success', '.alert-success', '.text-success', '.confirmation',
                    '.thank-you', '.submitted', '.complete', '[role="status"]'
                ];
                
                for (const selector of successSelectors) {
                    const elements = document.querySelectorAll(selector);
                    for (const element of elements) {
                        const text = element.textContent?.toLowerCase() || '';
                        if ((text.includes('success') || 
                             text.includes('thank you') || 
                             text.includes('submitted') ||
                             text.includes('complete')) &&
                            getComputedStyle(element).display !== 'none') {
                            return true;
                        }
                    }
                }
                
                // Check for URL change that might indicate success
                const url = window.location.href.toLowerCase();
                if (url.includes('success') || 
                    url.includes('thank-you') || 
                    url.includes('confirmation') ||
                    url.includes('complete')) {
                    return true;
                }
                
                return false;
            })();
            """
            
            return page.evaluate(js_code) or False
            
        except Exception as e:
            print(f"⚠️ Error detecting form success: {e}")
            return False
    
    def get_description(self, context: GoalContext) -> str:
        """
        Generate a detailed description of what this form fill goal is looking for.
        This analyzes the current page state and provides specific requirements.
        
        The description should include:
        - Goal statement and relevant field count
        - Detailed field-by-field status (completed vs pending)
        - Current values for filled fields
        - Form validation errors that need addressing
        - Submit/next button availability
        - Overall completion progress
        
        Format should be:
        ```
        Form fill goal: [description]
        Needs to fill X relevant fields:
          ✅ field_name (type): 'value' - COMPLETED
          ⏳ field_name (type): EMPTY - NEEDS FILLING
        Form has X validation errors:
          ❌ error_message
        Submit button is available and ready
        📊 Progress: X/Y fields completed (Z remaining)
        ```
        """
        page = context.page_reference
        if not page:
            return f"Form fill goal: {self.description} (no page reference available)"
        
        try:
            # Detect current form elements
            detected_fields = self.detect_elements_on_page(page, self.description)
            
            # Filter to get relevant fields for this goal
            relevant_fields = self._filter_relevant_fields(context, detected_fields)
            
            # Update field values for relevant fields only
            self._update_field_values(page, relevant_fields)
            
            # Detect form errors
            form_errors = self._detect_form_errors(page)
            
            # Detect submit/next buttons
            submit_info = self._detect_submit_and_next_buttons(page)
            
            # Count relevant form fields (input, select, textarea only)
            relevant_form_fields = [f for f in relevant_fields if f['tagName'] in ['input', 'select', 'textarea']]
            filled_relevant_fields = [f for f in relevant_form_fields if f.get('isFilled')]
            unfilled_relevant_fields = [f for f in relevant_form_fields if not f.get('isFilled')]
            
            # Build description based on current state
            description_parts = []
            
            # Main goal description
            description_parts.append(f"Form fill goal: {self.description}")
            
            # Trigger settings
            trigger_info = []
            if self.trigger_on_submit:
                trigger_info.append("submit")
            if self.trigger_on_field_input:
                trigger_info.append("field input")
            if trigger_info:
                description_parts.append(f"Triggers on: {', '.join(trigger_info)}")
            else:
                description_parts.append("Triggers: none (general evaluation only)")
            
            # Field requirements
            if relevant_form_fields:
                description_parts.append(f"Needs to fill {len(relevant_form_fields)} relevant fields:")
                
                for field in relevant_form_fields:
                    field_name = field.get('name', field.get('id', field.get('placeholder', 'unnamed')))
                    field_type = field.get('type', 'text')
                    is_filled = field.get('isFilled', False)
                    current_value = field.get('currentValue', '')
                    
                    if is_filled:
                        description_parts.append(f"  ✅ {field_name} ({field_type}): '{current_value}' - COMPLETED")
                    else:
                        description_parts.append(f"  ⏳ {field_name} ({field_type}): EMPTY - NEEDS FILLING")
            else:
                description_parts.append("No relevant form fields detected on current page")
            
            # Form errors
            if form_errors:
                description_parts.append(f"Form has {len(form_errors)} validation errors:")
                for error in form_errors[:3]:  # Show first 3 errors
                    description_parts.append(f"  ❌ {error}")
                if len(form_errors) > 3:
                    description_parts.append(f"  ... and {len(form_errors) - 3} more errors")
            
            # Submit button status
            if submit_info['submit_available']:
                description_parts.append("Submit button is available and ready")
            elif submit_info['next_available']:
                description_parts.append("Next/Continue button is available")
            else:
                description_parts.append("No submit or next button detected")
            
            # Completion status
            if relevant_form_fields:
                completion_ratio = len(filled_relevant_fields) / len(relevant_form_fields)
                if completion_ratio >= 1.0:
                    description_parts.append("🎯 ALL RELEVANT FIELDS COMPLETED - Goal can be marked as achieved")
                else:
                    remaining = len(unfilled_relevant_fields)
                    description_parts.append(f"📊 Progress: {len(filled_relevant_fields)}/{len(relevant_form_fields)} fields completed ({remaining} remaining)")
            
            result = "\n".join(description_parts)
            result += "\n\n"
            result += "Do NOT fill fields that are marked as COMPLETED or ✅"
            result += "\n\n"
            result += "If all the fields are marked as COMPLETED or ✅, and there is a submit button but it is not 'visible', return a scroll action"
            return result
            
        except Exception as e:
            return f"Form fill goal: {self.description} (error analyzing page: {e})"
