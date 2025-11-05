"""
Press Goal - Monitors keyboard press interactions to verify correct keys are pressed.
"""
from typing import Any, Dict, Optional, List

from pydantic import BaseModel, Field

from ai_utils import generate_model
from .base import BaseGoal, GoalResult, GoalStatus, EvaluationTiming, GoalContext, InteractionType


class KeyPressEvaluation(BaseModel):
    """Result of evaluating if pressed keys match the target description"""
    is_match: bool = Field(description="Whether the pressed keys match the target")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the match assessment")
    reasoning: str = Field(description="Explanation of why this match determination was made")


class PressGoal(BaseGoal):
    """
    Goal for pressing specific keys or key combinations.
    
    This goal monitors keyboard press interactions and determines if the correct
    keys were pressed based on the user's description.
    """
    
    # PressGoal should be evaluated BEFORE interaction to validate the planned key press
    EVALUATION_TIMING = EvaluationTiming.BEFORE
    
    def __init__(self, description: str, target_keys: str, **kwargs):
        super().__init__(description, **kwargs)
        self.target_keys = target_keys.lower().strip()
        self.planned_keys: Optional[str] = None
        self.actual_keys: Optional[str] = None
        self.press_attempted = False
        self.press_successful = False
        self.key_sequence: List[str] = []
    
    def on_interaction(self, interaction) -> None:
        """Track keyboard press interactions"""
        if interaction.interaction_type == InteractionType.PRESS:
            self.press_attempted = True
            self.actual_keys = getattr(interaction, 'keys_pressed', None)
            self.press_successful = interaction.success
            print(f"[PressGoal] Recorded key press: {self.actual_keys}")
    
    def record_planned_press(self, keys: str) -> None:
        """
        Record the keys that are planned to be pressed.
        This should be called BEFORE the press happens.
        """
        self.planned_keys = keys.lower().strip()
        print(f"[PressGoal] Recorded planned key press: {self.planned_keys}")
    
    def evaluate(self, context: GoalContext) -> GoalResult:
        """
        Evaluate if the correct keys are planned to be pressed (BEFORE timing).
        """
        # Check if we have planned interaction data (pre-interaction evaluation)
        planned_interaction = getattr(context, 'planned_interaction', None)
        if planned_interaction and planned_interaction.get('interaction_type') == InteractionType.PRESS:
            return self._evaluate_planned_press(context, planned_interaction)
        
        # Fallback to post-interaction evaluation if no planned interaction
        if not self.press_attempted:
            return GoalResult(
                status=GoalStatus.PENDING,
                confidence=1.0,
                reasoning="No key press interaction has been planned or attempted yet",
                next_actions=["Wait for key press interaction"]
            )
        
        if not self.press_successful:
            return GoalResult(
                status=GoalStatus.FAILED,
                confidence=0.9,
                reasoning="Key press interaction failed to execute",
                evidence={"error": "Key press execution failed"}
            )
        
        # Evaluate if the correct keys were pressed (fallback)
        base_knowledge = context.base_knowledge if hasattr(context, 'base_knowledge') else None
        return self._evaluate_key_match(self.actual_keys, self.target_keys, base_knowledge=base_knowledge)
    
    def _evaluate_planned_press(self, context: GoalContext, planned_interaction: Dict[str, Any]) -> GoalResult:
        """Evaluate a planned key press before it happens"""
        planned_keys = planned_interaction.get('keys_to_press', '')
        
        if not planned_keys:
            return GoalResult(
                status=GoalStatus.FAILED,
                confidence=0.0,
                reasoning="No keys specified in planned press interaction",
                evidence={"error": "Missing keys in planned interaction"}
            )
        
        # Record the planned press
        self.record_planned_press(planned_keys)
        
        # Evaluate if the planned keys match the target
        base_knowledge = context.base_knowledge if hasattr(context, 'base_knowledge') else None
        result = self._evaluate_key_match(planned_keys, self.target_keys, base_knowledge=base_knowledge)
        
        # For BEFORE timing, we want to validate the planned action
        # If keys match, return ACHIEVED (ready to proceed)
        # If keys don't match, return FAILED (should retry with correct keys)
        return result
    
    def _evaluate_key_match(self, actual_keys: str, target_keys: str, base_knowledge: Optional[List[str]] = None) -> GoalResult:
        """Evaluate if the actual keys match the target keys"""
        if not actual_keys:
            return GoalResult(
                status=GoalStatus.FAILED,
                confidence=0.0,
                reasoning="No keys were actually pressed",
                evidence={"error": "Missing actual keys"}
            )
        
        # Normalize both key strings for comparison
        actual_normalized = self._normalize_key_string(actual_keys)
        target_normalized = self._normalize_key_string(target_keys)
        
        # Build base knowledge section if provided
        base_knowledge_section = ""
        if base_knowledge:
            base_knowledge_section = "\n\nBASE KNOWLEDGE (Rules that guide evaluation):\n"
            for i, knowledge in enumerate(base_knowledge, 1):
                base_knowledge_section += f"{i}. {knowledge}\n"
            base_knowledge_section += "\nIMPORTANT: Apply these base knowledge rules when determining if keys match. They override general matching assumptions.\n"
        
        # Use AI to evaluate the match
        try:
            evaluation = generate_model(
                prompt=f"""
                Compare these two key press sequences:
                
                Target keys: "{target_keys}"
                Actual keys: "{actual_keys}"
                {base_knowledge_section}
                Determine if the actual keys match the target keys. Consider:
                1. Exact matches (e.g., "enter" = "enter")
                2. Equivalent keys (e.g., "return" = "enter", "cmd+c" = "ctrl+c" on different platforms)
                3. Key combinations (e.g., "ctrl+c" should match "control+c")
                4. Case insensitivity
                5. Whitespace differences
                6. Base knowledge rules (if provided above) that may make matching more lenient or specific
                
                Return your evaluation.
                """,
                model_object_type=KeyPressEvaluation,
            )
            
            if evaluation.is_match:
                return GoalResult(
                    status=GoalStatus.ACHIEVED,
                    confidence=evaluation.confidence,
                    reasoning=evaluation.reasoning,
                    evidence={
                        "target_keys": target_keys,
                        "actual_keys": actual_keys,
                        "match_evaluation": evaluation.model_dump()
                    }
                )
            else:
                return GoalResult(
                    status=GoalStatus.FAILED,
                    confidence=evaluation.confidence,
                    reasoning=evaluation.reasoning,
                    evidence={
                        "target_keys": target_keys,
                        "actual_keys": actual_keys,
                        "match_evaluation": evaluation.model_dump()
                    }
                )
                
        except Exception:
            # Fallback to simple string comparison
            is_match = actual_normalized == target_normalized
            confidence = 0.8 if is_match else 0.2
            
            return GoalResult(
                status=GoalStatus.ACHIEVED if is_match else GoalStatus.FAILED,
                confidence=confidence,
                reasoning=f"Simple string comparison: {'match' if is_match else 'no match'}",
                evidence={
                    "target_keys": target_keys,
                    "actual_keys": actual_keys,
                    "comparison_method": "fallback_string_comparison"
                }
            )
    
    def _normalize_key_string(self, key_string: str) -> str:
        """Normalize a key string for comparison"""
        if not key_string:
            return ""
        
        # Convert to lowercase and strip whitespace
        normalized = key_string.lower().strip()
        
        # Replace common variations
        replacements = {
            'return': 'enter',
            'cmd': 'ctrl',
            'command': 'ctrl',
            'control': 'ctrl',
            'alt': 'alt',
            'option': 'alt',
            'shift': 'shift',
            'space': ' ',
            'tab': '\t',
            'escape': 'esc',
            'backspace': 'backspace',
            'delete': 'del'
        }
        
        for old, new in replacements.items():
            normalized = normalized.replace(old, new)
        
        # Remove extra spaces and normalize separators
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    def get_description(self, context: GoalContext) -> str:
        """Generate a detailed description of what this goal is looking for"""
        status = "PENDING"
        if self.planned_keys:
            status = "VALIDATED" if self.planned_keys == self.target_keys else "MISMATCH"
        elif self.press_attempted:
            if self.press_successful:
                status = "COMPLETED" if self.actual_keys == self.target_keys else "FAILED"
            else:
                status = "FAILED"
        
        return f"""
Goal Type: Press Goal (BEFORE evaluation)
Target Keys: {self.target_keys}
Current Status: {status}
Requirements: Press the keys "{self.target_keys}"
Progress: {'✅ Keys validated and ready to press' if status == 'VALIDATED' else '⏳ Waiting for key press validation' if status == 'PENDING' else '❌ Key mismatch detected'}
Issues: {'None' if status in ['VALIDATED', 'COMPLETED'] else 'Key press not yet planned or incorrect keys planned'}
"""
