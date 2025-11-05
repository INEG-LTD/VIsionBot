"""
Form Fill Goal - Evaluates form field interactions before they happen.
"""
from __future__ import annotations

from typing import Dict, Any, Optional, Tuple

from pydantic import BaseModel, Field

from ai_utils import generate_model
from .base import BaseGoal, GoalResult, GoalStatus, GoalContext, EvaluationTiming


class ElementMatchEvaluation(BaseModel):
    """Result of evaluating if a form field matches the target description"""
    is_match: bool = Field(description="Whether the form field matches the target")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the match assessment")
    reasoning: str = Field(description="Explanation of why this match determination was made")


class FormFillGoal(BaseGoal):
    """
    Goal that evaluates form field interactions before they happen.
    
    This goal evaluates planned form interactions (type, select, upload) to ensure
    the correct form field is being targeted before the interaction occurs.
    """
    
    EVALUATION_TIMING = EvaluationTiming.BEFORE
    
    def __init__(
        self, 
        description: str,
        **kwargs
    ):
        super().__init__(description, **kwargs)
    
    def evaluate(self, context: GoalContext) -> GoalResult:
        """Evaluate form field interaction before it happens"""
        
        page = context.page_reference
        if not page:
            return GoalResult(
                status=GoalStatus.UNKNOWN,
                confidence=0.0,
                reasoning="No page reference available for form evaluation"
            )
        
        # Check if we have planned interaction data (pre-interaction evaluation)
        planned_interaction = getattr(context, 'planned_interaction', None)
        if planned_interaction and planned_interaction.get('interaction_type') in ['type', 'select', 'upload']:
            return self._evaluate_planned_form_interaction(context, planned_interaction)
        
        # No planned interaction - return pending
        return GoalResult(
            status=GoalStatus.PENDING,
            confidence=1.0,
            reasoning="No form interaction planned yet",
            next_actions=["Wait for form interaction to be planned"]
        )
    
    
    def _evaluate_element_match(
        self,
        actual_description: str,
        target_description: str,
        actual_element_info: Optional[Dict[str, Any]] = None,
        page_dimensions: Optional[Tuple[int, int]] = None,
        base_knowledge: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Use AI evaluation method for form field matching.
        Customized for form fields with specific context about field types, labels, and purpose.
        """
        try:
            print(f"[FormGoal] Evaluating form field match: {actual_description} vs {target_description}")
            
            # Enhanced system prompt with form field context
            element_context = ""
            if actual_element_info:
                element_context = f"""
                Form field details:
                - Type: {actual_element_info.get('elementType', 'unknown')}
                - Field type: {actual_element_info.get('type', 'text')}
                - Name: {actual_element_info.get('name', '')}
                - ID: {actual_element_info.get('id', '')}
                - Placeholder: {actual_element_info.get('placeholder', '')}
                - Label: {actual_element_info.get('ariaLabel', '')}
                - Text content: {actual_element_info.get('text', '')[:100]}
                - Required: {actual_element_info.get('required', False)}
                - Attributes: {str(actual_element_info.get('attributes', {}))[:200]}
                """
            
            # Build base knowledge section if provided
            base_knowledge_section = ""
            if base_knowledge:
                base_knowledge_section = "\n\nBASE KNOWLEDGE (Rules that guide evaluation):\n"
                for i, knowledge in enumerate(base_knowledge, 1):
                    base_knowledge_section += f"{i}. {knowledge}\n"
                base_knowledge_section += "\nIMPORTANT: Apply these base knowledge rules when determining if elements match. They override general matching assumptions.\n"
            
            system_prompt = f"""
            You are evaluating whether a form field matches the user's intent for form filling.

            User wants to fill: "{target_description}"
            Form field found: "{actual_description}"
            {element_context}
            {base_knowledge_section}
            Determine if these match semantically for form filling purposes. Consider:
            - Field names, labels, and placeholders (e.g., "email" could match "email address" or "e-mail")
            - Field types and purposes (e.g., "date" could match "birth date" or "appointment date")
            - Synonyms and similar terms (e.g., "name" vs "full name", "phone" vs "telephone")
            - Context and intent (e.g., "contact information" could match "phone number" field)
            - Form field functionality and purpose
            - Required vs optional field status
            - Base knowledge rules (if provided above) that may make matching more lenient or specific

            For form fields, be more flexible with matching since users often describe fields in general terms
            while the actual field might have more specific labels or names.

            Provide your evaluation with confidence and reasoning.
            """

            result = generate_model(
                prompt="Evaluate if the form field matches the target description for form filling.",
                model_object_type=ElementMatchEvaluation,
                system_prompt=system_prompt,
            )

            return result.model_dump()

        except Exception as e:
            print(f"AI evaluation failed: {e}, falling back to simple matching")
            return self._simple_string_match(actual_description, target_description)
    
    def _simple_string_match(self, actual: str, target: str) -> Dict[str, Any]:
        """Simple fallback matching when AI is not available"""
        actual_lower = actual.lower()
        target_lower = target.lower()
        
        # Exact match
        if actual_lower == target_lower:
            return {"is_match": True, "confidence": 1.0, "reasoning": "Exact match"}
        
        # Substring match
        if target_lower in actual_lower or actual_lower in target_lower:
            return {"is_match": True, "confidence": 0.8, "reasoning": "Substring match"}
        
        # Word overlap
        actual_words = set(actual_lower.split())
        target_words = set(target_lower.split())
        overlap = actual_words & target_words
        
        if overlap and len(overlap) >= len(target_words) * 0.5:
            return {
                "is_match": True, 
                "confidence": 0.6, 
                "reasoning": f"Word overlap: {overlap}"
            }
        
        return {"is_match": False, "confidence": 0.9, "reasoning": "No significant match found"}
    
    def _evaluate_planned_form_interaction(self, context: GoalContext, planned_interaction: Dict[str, Any]) -> GoalResult:
        """
        Evaluate a planned form interaction before it happens.
        This checks if the target form field matches the goal's requirements.
        """
        coordinates = planned_interaction.get('coordinates')
        if not coordinates:
            return GoalResult(
                status=GoalStatus.UNKNOWN,
                confidence=0.1,
                reasoning="No coordinates provided for planned form interaction"
            )
        
        x, y = coordinates
        interaction_type = planned_interaction.get('interaction_type', 'type')
        target_description = planned_interaction.get('target_description', '')
        
        print(f"[FormGoal] Evaluating planned {interaction_type} interaction at ({x}, {y}) for '{target_description}'")
        
        # Get element info at the planned coordinates
        try:
            element_info = self.get_element_info_at_coordinates(context.page_reference, x, y)
            if not element_info:
                # Request retry if we can't analyze the element
                if self.can_retry():
                    retry_reason = f"Could not analyze form field at coordinates ({x}, {y}) - element analysis failed"
                    if self.request_retry(retry_reason):
                        return GoalResult(
                            status=GoalStatus.PENDING,
                            confidence=0.3,
                            reasoning=f"Could not analyze form field at ({x}, {y}), requesting retry",
                            evidence={
                                "coordinates": (x, y),
                                "retry_requested": True,
                                "retry_count": self.retry_count
                            },
                            next_actions=["Retry plan generation to find valid form field"]
                        )
                
                return GoalResult(
                    status=GoalStatus.FAILED,
                    confidence=0.3,
                    reasoning=f"Could not analyze form field at ({x}, {y}) and max retries exceeded",
                    evidence={
                        "coordinates": (x, y),
                        "retry_count": self.retry_count,
                        "max_retries_exceeded": True
                    }
                )
            
            # Create element description for matching
            element_description = self._create_element_description(element_info)
            
            # Check if this is a form field
            if element_info.get('tagName', '').lower() not in ['input', 'select', 'textarea']:
                # Request retry if it's not a form field
                if self.can_retry():
                    retry_reason = f"Element at ({x}, {y}) is not a form field (tag: {element_info.get('tagName', 'unknown')}) - need to find form field"
                    if self.request_retry(retry_reason):
                        return GoalResult(
                            status=GoalStatus.PENDING,
                            confidence=0.9,
                            reasoning=f"Element at ({x}, {y}) is not a form field, requesting retry",
                            evidence={
                                "coordinates": (x, y),
                                "element_info": element_info,
                                "retry_requested": True,
                                "retry_count": self.retry_count
                            },
                            next_actions=["Retry plan generation to find form field"]
                        )
                
                return GoalResult(
                    status=GoalStatus.FAILED,
                    confidence=0.9,
                    reasoning=f"Element at ({x}, {y}) is not a form field and max retries exceeded",
                    evidence={
                        "coordinates": (x, y),
                        "element_info": element_info,
                        "retry_count": self.retry_count,
                        "max_retries_exceeded": True
                    }
                )
            
            # Evaluate if the form field matches the target description
            page_dims = (
                context.current_state.page_width,
                context.current_state.page_height,
            ) if context.current_state else None
            
            match_result = self._evaluate_element_match(
                element_description,
                target_description,
                element_info,
                page_dims,
                base_knowledge=context.base_knowledge if hasattr(context, 'base_knowledge') else None
            )
            
            if match_result["is_match"]:
                return GoalResult(
                    status=GoalStatus.ACHIEVED,
                    confidence=match_result["confidence"],
                    reasoning=f"Form field '{element_description}' matches target '{target_description}' - ready for interaction",
                    evidence={
                        "target_field": element_description,
                        "target_description": target_description,
                        "coordinates": coordinates,
                        "element_info": element_info,
                        "match_reasoning": match_result["reasoning"],
                        "evaluation_timing": "pre_interaction"
                    }
                )
            else:
                # Request retry if the form field doesn't match
                if self.can_retry():
                    retry_reason = f"Form field '{element_description}' does not match target '{target_description}' - {match_result['reasoning']}"
                    if self.request_retry(retry_reason):
                        return GoalResult(
                            status=GoalStatus.PENDING,
                            confidence=match_result["confidence"],
                            reasoning=f"Form field doesn't match target, requesting retry: {match_result['reasoning']}",
                            evidence={
                                "target_field": element_description,
                                "target_description": target_description,
                                "coordinates": (x, y),
                                "element_info": element_info,
                                "match_reasoning": match_result["reasoning"],
                                "retry_requested": True,
                                "retry_count": self.retry_count
                            },
                            next_actions=["Retry plan generation to find correct form field"]
                        )
                
                return GoalResult(
                    status=GoalStatus.FAILED,
                    confidence=match_result["confidence"],
                    reasoning=f"Form field '{element_description}' does not match target '{target_description}' and max retries exceeded - {match_result['reasoning']}",
                    evidence={
                        "target_field": element_description,
                        "target_description": target_description,
                        "coordinates": (x, y),
                        "element_info": element_info,
                        "match_reasoning": match_result["reasoning"],
                        "retry_count": self.retry_count,
                        "max_retries_exceeded": True
                    }
                )
                
        except Exception as e:
            return GoalResult(
                status=GoalStatus.UNKNOWN,
                confidence=0.1,
                reasoning=f"Error evaluating planned form interaction: {e}",
                evidence={"error": str(e)}
            )
    
    def _create_element_description(self, element_info: Dict[str, Any]) -> str:
        """Create a description of the form field element"""
        element_type = element_info.get('elementType', 'form field')
        field_name = element_info.get('name', '')
        field_id = element_info.get('id', '')
        placeholder = element_info.get('placeholder', '')
        label = element_info.get('ariaLabel', '')
        field_type = element_info.get('type', 'text')
        
        # Build description from available information
        description_parts = [element_type]
        
        if field_name:
            description_parts.append(f"named '{field_name}'")
        elif field_id:
            description_parts.append(f"with id '{field_id}'")
        elif placeholder:
            description_parts.append(f"with placeholder '{placeholder}'")
        elif label:
            description_parts.append(f"labeled '{label}'")
        
        if field_type and field_type != 'text':
            description_parts.append(f"of type '{field_type}'")
        
        return " ".join(description_parts)
    
    def get_element_info_at_coordinates(self, page, x: int, y: int) -> Optional[Dict[str, Any]]:
        """Get element information at the specified coordinates"""
        try:
            # Use JavaScript to get element info at coordinates
            js_code = f"""
            (function() {{
                const element = document.elementFromPoint({x}, {y});
                if (!element) return null;
                
                const rect = element.getBoundingClientRect();
                const computedStyle = getComputedStyle(element);
                
                return {{
                    tagName: element.tagName.toLowerCase(),
                    elementType: element.tagName.toLowerCase(),
                    type: element.type || '',
                    name: element.name || '',
                    id: element.id || '',
                    placeholder: element.placeholder || '',
                    ariaLabel: element.getAttribute('aria-label') || '',
                    className: element.className || '',
                    textContent: element.textContent || '',
                    value: element.value || '',
                    required: element.required || false,
                    isClickable: computedStyle.pointerEvents !== 'none' && 
                                computedStyle.display !== 'none' && 
                                computedStyle.visibility !== 'hidden',
                    coordinates: {{
                        x: Math.round(rect.left + rect.width / 2),
                        y: Math.round(rect.top + rect.height / 2)
                    }},
                    attributes: {{
                        type: element.type,
                        name: element.name,
                        id: element.id,
                        placeholder: element.placeholder,
                        'aria-label': element.getAttribute('aria-label'),
                        class: element.className
                    }}
                }};
            }})();
            """
            
            return page.evaluate(js_code)
            
        except Exception as e:
            print(f"Error getting element info at coordinates ({x}, {y}): {e}")
            return None
    
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
