"""
Select Goal - Validates select dropdown fields before selection.
"""
from typing import Dict, Any

from .form_field_goal import FormFieldGoal
from .base import GoalResult, GoalStatus, GoalContext


class SelectGoal(FormFieldGoal):
    """
    Goal for selecting options in dropdown fields.
    Validates that the correct select field is targeted before selection.
    """
    
    def evaluate(self, context: GoalContext) -> GoalResult:
        """Evaluate select field before selection"""
        page = context.page_reference
        if not page:
            return GoalResult(
                status=GoalStatus.UNKNOWN,
                confidence=0.0,
                reasoning="No page reference available"
            )
        
        planned_interaction = getattr(context, 'planned_interaction', None)
        if not planned_interaction or planned_interaction.get('interaction_type') != 'select':
            return GoalResult(
                status=GoalStatus.PENDING,
                confidence=1.0,
                reasoning="No select interaction planned yet"
            )
        
        return self._evaluate_form_field(context, planned_interaction, ['select'], "SelectGoal")
    
    def _validate_element_type(self, element_info: Dict[str, Any], expected_types: list) -> bool:
        """Check if element is a select field (can be <select> or custom dropdown)"""
        tag_name = element_info.get('tagName', '').lower()
        attributes = element_info.get('attributes', {})
        role = attributes.get('role', '').lower() if attributes else ''
        
        # Traditional select element
        if tag_name == 'select':
            return True
        
        # Custom dropdown with role indicators
        if role in ['combobox', 'listbox']:
            return True
        
        return False
    
    def _evaluate_form_field(self, context: GoalContext, planned_interaction: Dict[str, Any], expected_types: list, goal_name: str) -> GoalResult:
        """Override to use custom validation for select fields"""
        coordinates = planned_interaction.get('coordinates')
        if not coordinates:
            return GoalResult(
                status=GoalStatus.UNKNOWN,
                confidence=0.1,
                reasoning="No coordinates provided"
            )
        
        x, y = coordinates
        target_description = planned_interaction.get('target_description', self.target_description)
        
        print(f"[{goal_name}] Evaluating planned select interaction at ({x}, {y}) for '{target_description}'")
        
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
            
            # Validate it's a select field (using custom validation)
            if not self._validate_element_type(element_info, expected_types):
                if self.can_retry():
                    if self.request_retry(f"Element at ({x}, {y}) is not a select field"):
                        return GoalResult(
                            status=GoalStatus.PENDING,
                            confidence=0.9,
                            reasoning=f"Element at ({x}, {y}) is not a select field, requesting retry",
                            next_actions=["Retry plan generation"]
                        )
                return GoalResult(
                    status=GoalStatus.FAILED,
                    confidence=0.9,
                    reasoning=f"Element at ({x}, {y}) is not a select field"
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
                    reasoning=f"Select field '{element_description}' matches target '{target_description}'",
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
                reasoning=f"Error evaluating select interaction: {e}"
            )
    
    def get_description(self, context: GoalContext) -> str:
        """Generate a detailed description of what this goal is looking for"""
        planned_interaction = getattr(context, 'planned_interaction', None)
        status = "PENDING"
        
        if planned_interaction and planned_interaction.get('interaction_type') == 'select':
            # Check if we have evaluation result
            if hasattr(self, '_last_evaluation') and self._last_evaluation:
                if self._last_evaluation.status == GoalStatus.ACHIEVED:
                    status = "VALIDATED"
                elif self._last_evaluation.status == GoalStatus.FAILED:
                    status = "FAILED"
        
        return f"""
Goal Type: Select Goal (BEFORE evaluation)
Description: {self.description}
Target Field: {self.target_description}
Current Status: {status}
Requirements: Select from the dropdown field matching "{self.target_description}"
Progress: {'✅ Field validated and ready for selection' if status == 'VALIDATED' else '⏳ Waiting for select interaction validation' if status == 'PENDING' else '❌ Field validation failed'}
Issues: {'None' if status == 'VALIDATED' else 'Select interaction not yet planned or field mismatch detected'}
"""
