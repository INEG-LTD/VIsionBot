"""
Scroll Goal - Simple, clean implementation for scroll actions.
"""
import re
from typing import Optional

from pydantic import BaseModel, Field

from ai_utils import generate_text

from .base import BaseGoal, GoalResult, GoalStatus, EvaluationTiming, GoalContext, InteractionType


class ScrollInterpretation(BaseModel):
    """Simple interpretation of a scroll request"""
    target_x: int = Field(description="Target scroll X position in pixels")
    target_y: int = Field(description="Target scroll Y position in pixels")
    direction: str = Field(description="Scroll direction: up, down, left, right")
    axis: str = Field(description="Scroll axis: vertical or horizontal")


class ScrollGoal(BaseGoal):
    """
    Simple goal for scrolling by specific amounts.
    
    Interprets user requests like "scroll down", "scroll to bottom", "scroll left 200px"
    and validates that the actual scroll matches the request.
    """
    
    EVALUATION_TIMING = EvaluationTiming.BEFORE
    
    def __init__(self, description: str, user_request: str, **kwargs):
        super().__init__(description, needs_detection=False, **kwargs)
        self.user_request = user_request.lower().strip()
        self.interpreted_scroll: Optional[ScrollInterpretation] = None
    
    def interpret_request(self, context: GoalContext) -> ScrollInterpretation:
        """Interpret the user's scroll request into a concrete scroll action using AI"""
        if self.interpreted_scroll:
            return self.interpreted_scroll
            
        state = context.current_state
        request = self.user_request
        
        try:
            
            prompt = f"""
You are a scroll interpretation assistant. Given a user's scroll request and current page state, determine the exact target scroll position.

User Request: "{request}"

Current Page State:
- Page Height: {state.page_height}px
- Page Width: {state.page_width}px  
- Current Scroll Y: {state.scroll_y}px
- Current Scroll X: {state.scroll_x}px

Interpret the scroll request and respond with a JSON object containing:
- target_x: target X scroll position in pixels
- target_y: target Y scroll position in pixels
- direction: "up", "down", "left", or "right"
- axis: "vertical" or "horizontal"

Examples:
- "scroll down" → {{"target_x": {state.scroll_x}, "target_y": {min(state.scroll_y + 300, state.page_height)}, "direction": "down", "axis": "vertical"}}
- "scroll to bottom" → {{"target_x": {state.scroll_x}, "target_y": {state.page_height}, "direction": "down", "axis": "vertical"}}
- "scroll right 200px" → {{"target_x": {min(state.scroll_x + 200, state.page_width)}, "target_y": {state.scroll_y}, "direction": "right", "axis": "horizontal"}}
- "scroll down a page" → {{"target_x": {state.scroll_x}, "target_y": {min(state.scroll_y + state.page_height, state.page_height)}, "direction": "down", "axis": "vertical"}}
- "scroll to middle" → {{"target_x": {state.scroll_x}, "target_y": {int(state.page_height / 2)}, "direction": {"down" if int(state.page_height / 2) > state.scroll_y else "up"}, "axis": "vertical"}}
- "scroll left a bit" → {{"target_x": {max(state.scroll_x - 100, 0)}, "target_y": {state.scroll_y}, "direction": "left", "axis": "horizontal"}}

Respond with only the JSON object, no other text.
"""
            
            response = generate_text(
                prompt=prompt,
                reasoning_level="minimal",
                system_prompt="You are a scroll interpretation assistant. Given a user's scroll request and current page state, determine the exact scroll action needed.",
                model="gpt-5-nano"
            )
            
            # Parse the JSON response
            import json
            result = json.loads(response.strip())
            
            self.interpreted_scroll = ScrollInterpretation(
                target_x=result["target_x"],
                target_y=result["target_y"],
                direction=result["direction"],
                axis=result["axis"]
            )
            
            return self.interpreted_scroll
            
        except Exception as e:
            print(f"[ScrollGoal] AI interpretation failed: {e}, using fallback")
            # Fallback to simple interpretation
            return self._fallback_interpretation(state, request)
    
    def _fallback_interpretation(self, state, request: str) -> ScrollInterpretation:
        """Simple fallback interpretation when AI fails"""
        # Default values
        target_x = state.scroll_x
        target_y = state.scroll_y
        direction = "down"
        axis = "vertical"
        
        # Basic direction detection
        if any(word in request for word in ["left", "right"]):
            axis = "horizontal"
            direction = "left" if "left" in request else "right"
        elif any(word in request for word in ["up", "down"]):
            axis = "vertical"
            direction = "up" if "up" in request else "down"
        
        # Calculate target positions based on request
        if "px" in request:
            pixel_match = re.search(r'(\d+)\s*px', request)
            if pixel_match:
                amount = int(pixel_match.group(1))
                if axis == "horizontal":
                    if direction == "right":
                        target_x = min(state.scroll_x + amount, state.page_width)
                    else:  # left
                        target_x = max(state.scroll_x - amount, 0)
                else:  # vertical
                    if direction == "down":
                        target_y = min(state.scroll_y + amount, state.page_height)
                    else:  # up
                        target_y = max(state.scroll_y - amount, 0)
        elif "to bottom" in request:
            target_y = state.page_height
        elif "to top" in request:
            target_y = 0
        elif any(word in request for word in ["bit", "slightly", "little"]):
            amount = 100
            if axis == "horizontal":
                if direction == "right":
                    target_x = min(state.scroll_x + amount, state.page_width)
                else:  # left
                    target_x = max(state.scroll_x - amount, 0)
            else:  # vertical
                if direction == "down":
                    target_y = min(state.scroll_y + amount, state.page_height)
                else:  # up
                    target_y = max(state.scroll_y - amount, 0)
        else:
            # Default scroll down 300px
            target_y = min(state.scroll_y + 300, state.page_height)
        
        self.interpreted_scroll = ScrollInterpretation(
            target_x=target_x,
            target_y=target_y,
            direction=direction,
            axis=axis
        )
        
        return self.interpreted_scroll
    
    def evaluate(self, context: GoalContext) -> GoalResult:
        """Evaluate if the scroll goal has been achieved"""
        if not self.interpreted_scroll:
            self.interpreted_scroll = self.interpret_request(context)
            
        planned_interaction = getattr(context, 'planned_interaction', None)
        tolerance = 50  # 50px tolerance for position matching
        if planned_interaction and planned_interaction.get('interaction_type') == InteractionType.SCROLL:
            target_x = planned_interaction.get('target_x')
            target_y = planned_interaction.get('target_y')
            scroll_direction = planned_interaction.get('scroll_direction')
            scroll_axis = planned_interaction.get('scroll_axis')
            
            required_target_x = self.interpreted_scroll.target_x
            required_target_y = self.interpreted_scroll.target_y
            required_scroll_direction = self.interpreted_scroll.direction
            required_scroll_axis = self.interpreted_scroll.axis
            
            x_match = abs(target_x - required_target_x) <= tolerance
            y_match = abs(target_y - required_target_y) <= tolerance
            direction_match = scroll_direction == required_scroll_direction
            axis_match = scroll_axis == required_scroll_axis
            
            if not x_match:
                return GoalResult(
                    status=GoalStatus.FAILED,
                    confidence=0.8,
                    reasoning=f"Target X position mismatch: planned {target_x}px, required {required_target_x}px"
                )
            if not y_match:
                return GoalResult(
                    status=GoalStatus.FAILED,
                    confidence=0.8,
                    reasoning=f"Target Y position mismatch: planned {target_y}px, required {required_target_y}px"
                )
            if not direction_match:
                return GoalResult(
                    status=GoalStatus.FAILED,
                    confidence=0.8,
                    reasoning=f"Scroll direction mismatch: planned {scroll_direction}, required {required_scroll_direction}"
                )
            if not axis_match:
                return GoalResult(
                    status=GoalStatus.FAILED,
                    confidence=0.8,
                    reasoning=f"Scroll axis mismatch: planned {scroll_axis}, required {required_scroll_axis}"
                )
                
            return GoalResult(
                status=GoalStatus.ACHIEVED,
                confidence=0.9,
                reasoning=f"Scroll planned correctly: to ({target_x}, {target_y}) {scroll_direction} ({scroll_axis})"
            )
        
        return GoalResult(
            status=GoalStatus.FAILED,
            confidence=0.5,
            reasoning="No planned scroll interaction found"
        )
    
    def get_description(self, context: GoalContext) -> str:
        """Generate description with calculated scroll requirements"""
        if not self.interpreted_scroll:
            self.interpreted_scroll = self.interpret_request(context)
        
        scroll = self.interpreted_scroll
        
        return f"""
                    Goal Type: Scroll Goal
                    User Request: {self.user_request}
                    Required Action: Scroll to position ({scroll.target_x}, {scroll.target_y}) {scroll.direction} ({scroll.axis})
                    Status: {'✅ Ready' if self.interpreted_scroll else '⏳ Waiting for planning'}
                """