"""
Date Goal - Validates date/time input fields before filling.
"""
from typing import Dict, Any

from .form_field_goal import FormFieldGoal
from .base import GoalResult, GoalStatus, GoalContext


class DateGoal(FormFieldGoal):
    """
    Goal for filling date/time input fields.
    Validates that the correct date field is targeted before filling.
    """
    
    def evaluate(self, context: GoalContext) -> GoalResult:
        """Evaluate date/time input field before filling"""
        page = context.page_reference
        if not page:
            return GoalResult(
                status=GoalStatus.UNKNOWN,
                confidence=0.0,
                reasoning="No page reference available"
            )
        
        planned_interaction = getattr(context, 'planned_interaction', None)
        if not planned_interaction or planned_interaction.get('interaction_type') != 'datetime':
            return GoalResult(
                status=GoalStatus.PENDING,
                confidence=1.0,
                reasoning="No datetime interaction planned yet"
            )
        
        expected_types = [
            ('input', 'date'),
            ('input', 'time'),
            ('input', 'datetime-local'),
            ('input', 'month'),
            ('input', 'week')
        ]
        
        return self._evaluate_form_field(context, planned_interaction, expected_types, "DateGoal")
    
    def get_description(self, context: GoalContext) -> str:
        """Generate a detailed description of what this goal is looking for"""
        planned_interaction = getattr(context, 'planned_interaction', None)
        status = "PENDING"
        
        if planned_interaction and planned_interaction.get('interaction_type') == 'datetime':
            # Check if we have evaluation result
            if hasattr(self, '_last_evaluation') and self._last_evaluation:
                if self._last_evaluation.status == GoalStatus.ACHIEVED:
                    status = "VALIDATED"
                elif self._last_evaluation.status == GoalStatus.FAILED:
                    status = "FAILED"
        
        return f"""
Goal Type: Date Goal (BEFORE evaluation)
Description: {self.description}
Target Field: {self.target_description}
Current Status: {status}
Requirements: Fill the date/time field matching "{self.target_description}"
Progress: {'✅ Field validated and ready for date input' if status == 'VALIDATED' else '⏳ Waiting for datetime interaction validation' if status == 'PENDING' else '❌ Field validation failed'}
Issues: {'None' if status == 'VALIDATED' else 'Datetime interaction not yet planned or field mismatch detected'}
"""
