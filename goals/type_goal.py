"""
Type Goal - Validates text input fields before typing.
"""
from typing import Dict, Any

from .form_field_goal import FormFieldGoal
from .base import GoalResult, GoalStatus, GoalContext


class TypeGoal(FormFieldGoal):
    """
    Goal for typing into text input fields.
    Validates that the correct text field is targeted before typing.
    """
    
    def __init__(self, description: str, target_description: str, **kwargs):
        super().__init__(description, target_description, **kwargs)
    
    def evaluate(self, context: GoalContext) -> GoalResult:
        """Evaluate text input field before typing"""
        page = context.page_reference
        if not page:
            return GoalResult(
                status=GoalStatus.UNKNOWN,
                confidence=0.0,
                reasoning="No page reference available"
            )
        
        planned_interaction = getattr(context, 'planned_interaction', None)
        if not planned_interaction or planned_interaction.get('interaction_type') != 'type':
            return GoalResult(
                status=GoalStatus.PENDING,
                confidence=1.0,
                reasoning="No type interaction planned yet"
            )
        
        expected_types = [
            ('input', 'text'),
            ('input', 'email'),
            ('input', 'password'),
            ('input', 'number'),
            ('input', 'search'),
            ('input', 'url'),
            ('input', 'tel'),
            'textarea'
        ]
        
        return self._evaluate_form_field(context, planned_interaction, expected_types, "TypeGoal")
    
    def get_description(self, context: GoalContext) -> str:
        """Generate a detailed description of what this goal is looking for"""
        planned_interaction = getattr(context, 'planned_interaction', None)
        status = "PENDING"
        
        if planned_interaction and planned_interaction.get('interaction_type') == 'type':
            # Check if we have evaluation result
            if hasattr(self, '_last_evaluation') and self._last_evaluation:
                if self._last_evaluation.status == GoalStatus.ACHIEVED:
                    status = "VALIDATED"
                elif self._last_evaluation.status == GoalStatus.FAILED:
                    status = "FAILED"
        
        return f"""
Goal Type: Type Goal (BEFORE evaluation)
Description: {self.description}
Target Field: {self.target_description}
Current Status: {status}
Requirements: Type into the text field matching "{self.target_description}"
Progress: {'✅ Field validated and ready for typing' if status == 'VALIDATED' else '⏳ Waiting for type interaction validation' if status == 'PENDING' else '❌ Field validation failed'}
Issues: {'None' if status == 'VALIDATED' else 'Type interaction not yet planned or field mismatch detected'}
"""
