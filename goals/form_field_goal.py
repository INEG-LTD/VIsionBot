"""
Base Form Field Goal - Shared logic for form field validation goals.
"""
from __future__ import annotations

from typing import Dict, Any, Optional, List

from pydantic import BaseModel, Field

from ai_utils import generate_model
from .base import BaseGoal, GoalResult, GoalStatus, GoalContext, EvaluationTiming


class ElementMatchEvaluation(BaseModel):
    """Result of evaluating if a form field matches the target description"""
    is_match: bool = Field(description="Whether the form field matches the target")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the match assessment")
    reasoning: str = Field(description="Explanation of why this match determination was made")


class FormFieldGoal(BaseGoal):
    """
    Base class for form field validation goals.
    Provides shared logic for element matching and validation.
    """
    
    EVALUATION_TIMING = EvaluationTiming.BEFORE
    
    def __init__(self, description: str, target_description: str, **kwargs):
        super().__init__(description, **kwargs)
        self.target_description = target_description
    
    def get_element_info_at_coordinates(self, page, x: int, y: int) -> Optional[Dict[str, Any]]:
        """Get element information at the specified coordinates"""
        try:
            js_code = f"""
            (function() {{
                const element = document.elementFromPoint({x}, {y});
                if (!element) return null;
                
                const rect = element.getBoundingClientRect();
                const computedStyle = getComputedStyle(element);

                const editableAncestor = element.closest('[contenteditable="true"]');
                let editableAncestorTag = null;
                let editableAncestorRole = null;
                if (editableAncestor) {{
                    editableAncestorTag = editableAncestor.tagName ? editableAncestor.tagName.toLowerCase() : null;
                    editableAncestorRole = editableAncestor.getAttribute('role') || '';
                }}

                let docEditableSurface = false;
                try {{
                    if (!editableAncestor) {{
                        if (element.tagName && element.tagName.toLowerCase() === 'canvas') {{
                            const iframe = element.closest('iframe');
                            if (iframe && iframe.contentDocument) {{
                                const iframeActive = iframe.contentDocument.activeElement;
                                if (iframeActive && (iframeActive.isContentEditable || iframeActive.getAttribute('contenteditable') === 'true')) {{
                                    docEditableSurface = true;
                                }}
                            }}
                        }}
                        const active = document.activeElement;
                        if (active && (active.isContentEditable || active.getAttribute('contenteditable') === 'true')) {{
                            docEditableSurface = docEditableSurface || active.contains(element) || element.contains(active);
                        }}
                    }}
                }} catch (err) {{
                    docEditableSurface = docEditableSurface || false;
                }}
                
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
                    role: element.getAttribute('role') || '',
                    isContentEditable: !!(element.isContentEditable || element.getAttribute('contenteditable') === 'true'),
                    hasEditableAncestor: !!editableAncestor,
                    editableAncestorTag: editableAncestorTag,
                    editableAncestorRole: editableAncestorRole,
                    docEditableSurface: docEditableSurface,
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
                        class: element.className,
                        role: element.getAttribute('role'),
                        contenteditable: element.getAttribute('contenteditable')
                    }}
                }};
            }})();
            """
            
            return page.evaluate(js_code)
            
        except Exception as e:
            print(f"Error getting element info at coordinates ({x}, {y}): {e}")
            return None
    
    def _create_element_description(self, element_info: Dict[str, Any]) -> str:
        """Create a description of the form field element"""
        element_type = element_info.get('elementType', 'form field')
        field_name = element_info.get('name', '')
        field_id = element_info.get('id', '')
        placeholder = element_info.get('placeholder', '')
        label = element_info.get('ariaLabel', '')
        field_type = element_info.get('type', 'text')
        
        description_parts = [element_type]
        
        if field_name:
            description_parts.append(f"named '{field_name}'")
        elif field_id:
            description_parts.append(f"with id '{field_id}'")
        elif placeholder:
            description_parts.append(f"with placeholder '{placeholder}'")
        elif label:
            description_parts.append(f"labeled '{label}'")
        
        if(bool(element_info.get('isContentEditable'))):
            description_parts.append("(content editable)")
        if field_type and field_type != 'text':
            description_parts.append(f"of type '{field_type}'")
        if element_info.get('role'):
            description_parts.append(f"(role='{element_info.get('role')}')")
        
        return " ".join(description_parts)
    
    def _evaluate_element_match(
        self,
        actual_description: str,
        target_description: str,
        actual_element_info: Optional[Dict[str, Any]] = None,
        base_knowledge: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Use LLM to evaluate if form field matches target description"""
        try:
            print(f"[FormFieldGoal] Evaluating match: {actual_description} vs {target_description}")
            
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
                - Required: {actual_element_info.get('required', False)}
                """
            
            # Build base knowledge section if provided
            base_knowledge_section = ""
            if base_knowledge:
                base_knowledge_section = "\n\nBASE KNOWLEDGE (Rules that guide evaluation):\n"
                for i, knowledge in enumerate(base_knowledge, 1):
                    base_knowledge_section += f"{i}. {knowledge}\n"
                base_knowledge_section += "\nIMPORTANT: Apply these base knowledge rules when determining if elements match. They override general matching assumptions.\n"
            
            system_prompt = f"""
            You are evaluating whether a form field matches the user's intent.

            User wants to interact with: "{target_description}"
            Form field found: "{actual_description}"
            {element_context}
            {base_knowledge_section}
            Determine if these match semantically. Consider:
            - Field names, labels, placeholders (e.g., "email" matches "email address")
            - Field types and purposes (e.g., "date" matches "birth date" or "appointment date")
            - Synonyms and similar terms (e.g., "name" vs "full name", "phone" vs "telephone")
            - Context and intent
            - Base knowledge rules (if provided above) that may make matching more lenient or specific

            Return your evaluation with confidence and reasoning.
            """

            result = generate_model(
                prompt="Evaluate if the form field matches the target description.",
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
        
        if actual_lower == target_lower:
            return {"is_match": True, "confidence": 1.0, "reasoning": "Exact match"}
        
        if target_lower in actual_lower or actual_lower in target_lower:
            return {"is_match": True, "confidence": 0.8, "reasoning": "Substring match"}
        
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
    
    def _validate_element_type(self, element_info: Dict[str, Any], expected_types: list) -> bool:
        """Check if element matches expected types"""
        tag_name = element_info.get('tagName', '').lower()
        input_type = element_info.get('type', '').lower()
        role = element_info.get('role', '')
        if role:
            role = role.lower()
        is_content_editable = bool(element_info.get('isContentEditable'))
        
        for expected in expected_types:
            if isinstance(expected, tuple):
                # Check both tag and type (e.g., ('input', 'text'))
                if tag_name == expected[0] and input_type == expected[1]:
                    return True
            else:
                # Check tag only (e.g., 'select', 'textarea')
                expected_lower = str(expected).lower()
                if expected_lower == 'contenteditable' and is_content_editable:
                    return True
                if expected_lower.startswith('role:') and role == expected_lower.split(':', 1)[1]:
                    return True
                if tag_name == expected_lower:
                    return True
        
        # Fallback heuristics for rich editors (e.g., Google Docs)
        has_editable_ancestor = bool(element_info.get('hasEditableAncestor'))
        if has_editable_ancestor:
            return True
        if element_info.get('docEditableSurface'):
            return True
        if is_content_editable and any(isinstance(exp, str) and exp.lower() == 'contenteditable' for exp in expected_types):
            return True
        
        return False
    
    def _evaluate_form_field(self, context: GoalContext, planned_interaction: Dict[str, Any], expected_types: list, goal_name: str) -> GoalResult:
        """Common evaluation logic for form field interactions"""
        coordinates = planned_interaction.get('coordinates')
        if not coordinates:
            return GoalResult(
                status=GoalStatus.UNKNOWN,
                confidence=0.1,
                reasoning="No coordinates provided"
            )
        
        x, y = coordinates
        target_description = planned_interaction.get('target_description', self.target_description)
        
        print(f"[{goal_name}] Evaluating planned interaction at ({x}, {y}) for '{target_description}'")
        
        try:
            element_info = self.get_element_info_at_coordinates(context.page_reference, x, y)
            if not element_info:
                if self.can_retry():
                    if self.request_retry(f"Could not analyze field at ({x}, {y})"):
                        return GoalResult(
                            status=GoalStatus.PENDING,
                            confidence=0.3,
                            reasoning=f"Could not analyze field at ({x}, {y}), requesting retry",
                            next_actions=["Retry plan generation"]
                        )
                return GoalResult(
                    status=GoalStatus.FAILED,
                    confidence=0.3,
                    reasoning=f"Could not analyze field at ({x}, {y})"
                )
            
            # Validate element type
            if not self._validate_element_type(element_info, expected_types):
                type_str = ', '.join([str(t) for t in expected_types])
                if self.can_retry():
                    if self.request_retry(f"Element at ({x}, {y}) is not the expected field type ({type_str})"):
                        return GoalResult(
                            status=GoalStatus.PENDING,
                            confidence=0.9,
                            reasoning=f"Element at ({x}, {y}) is not the expected field type, requesting retry",
                            next_actions=["Retry plan generation"]
                        )
                return GoalResult(
                    status=GoalStatus.FAILED,
                    confidence=0.9,
                    reasoning=f"Element at ({x}, {y}) is not the expected field type"
                )
            
            # Check if element matches target description
            element_description = self._create_element_description(element_info)
            match_result = self._evaluate_element_match(
                element_description,
                target_description,
                element_info,
                base_knowledge=context.base_knowledge if hasattr(context, 'base_knowledge') else None
            )
            
            if match_result["is_match"]:
                return GoalResult(
                    status=GoalStatus.ACHIEVED,
                    confidence=match_result["confidence"],
                    reasoning=f"Field '{element_description}' matches target '{target_description}'",
                    evidence={
                        "target_field": element_description,
                        "target_description": target_description,
                        "match_reasoning": match_result["reasoning"]
                    }
                )
            else:
                if self.can_retry():
                    if self.request_retry(f"Field '{element_description}' does not match '{target_description}'"):
                        return GoalResult(
                            status=GoalStatus.PENDING,
                            confidence=match_result["confidence"],
                            reasoning=f"Field doesn't match target, requesting retry: {match_result['reasoning']}",
                            next_actions=["Retry plan generation"]
                        )
                return GoalResult(
                    status=GoalStatus.FAILED,
                    confidence=match_result["confidence"],
                    reasoning=f"Field '{element_description}' does not match '{target_description}' - {match_result['reasoning']}"
                )
                
        except Exception as e:
            return GoalResult(
                status=GoalStatus.UNKNOWN,
                confidence=0.1,
                reasoning=f"Error evaluating form field interaction: {e}"
            )
