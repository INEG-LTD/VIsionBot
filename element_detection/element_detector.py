"""
Detects UI elements using AI vision models with numbered overlays.
"""
from typing import List, Dict, Any, Optional

from ai_utils import generate_model
from models import PageElements, DetectedElement, PageInfo
from vision_utils import clamp_coordinate


class ElementDetector:
    """Detects UI elements using AI vision models"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    def detect_elements_with_overlays(
        self,
        goal_description: str,
        additional_context: str,
        screenshot: bytes,
        element_data: List[Dict[str, Any]],
        page_info: PageInfo
    ) -> Optional[PageElements]:
        """Detect relevant UI elements using numbered overlays for precise coordinate mapping"""
        
        print(f"Detecting elements with overlays for goal: {goal_description}\n")
        
        # Create a summary of available elements for the AI
        element_summary = []
        for elem in element_data:
            element_summary.append(f"Element #{elem['index']}: {elem['description']}")
        
        system_prompt = f"""
        You are an expert web automation assistant analyzing a screenshot with numbered element overlays.

        ## Context
        - Current Page URL: {page_info.url}
        - User's Goal: {goal_description}
        {f"- Additional Information: {additional_context}" if additional_context else ""}

        ## Available Elements
        The screenshot shows numbered red overlays (1, 2, 3, etc.) on interactive elements:
        {chr(10).join(element_summary)}

        ## Instructions
        1. Look at the screenshot and identify which numbered elements are most relevant to achieving the user's goal
        2. Focus on elements that directly help accomplish the goal
        3. Return information about the relevant elements using their exact overlay numbers

        ## Output Schema
        Return a JSON list where each object represents a relevant element with:
        - "element_type": Type of UI element ("button", "input", "link", "select", etc.)
        - "description": What this element does in relation to the goal
        - "box_2d": Use the EXACT coordinates I provide below for the overlay number you select
        - "confidence": Your confidence (0.0-1.0) this element helps achieve the goal
        - "is_clickable": Whether the element is clickable
        - "field_subtype": For inputs, specify subtype ("text", "email", "password", etc.) or null
        - "requires_special_handling": true for selects, uploads, date/time fields
        - "overlay_number": The red overlay number from the screenshot

        ## Element Coordinates
        Use these EXACT coordinates based on overlay numbers:
        {chr(10).join([f"Overlay #{elem['index']}: {elem['normalizedCoords']}" for elem in element_data])}

        CRITICAL: You must use the exact coordinates provided above based on the overlay number you select.
        """
        
        user_prompt = f"Analyze the numbered overlays and identify which elements are most relevant for: {goal_description}"
        
        try:
            # Custom response parsing since we need to map overlay numbers to coordinates
            response = generate_model(
                prompt=user_prompt,
                model_object_type=PageElements,
                reasoning_level="none",
                system_prompt=system_prompt,
                model=self.model_name,
                image=screenshot
            )
            
            if response and response.elements:
                # Ensure coordinates are properly mapped from overlay data
                for element in response.elements:
                    # Find matching overlay number if provided
                    overlay_num = getattr(element, 'overlay_number', None)
                    if overlay_num:
                        # Find the element data for this overlay number
                        matching_data = next((elem for elem in element_data if elem['index'] == overlay_num), None)
                        if matching_data:
                            element.box_2d = matching_data['normalizedCoords']
                            print(f"  ✅ Mapped overlay #{overlay_num} to coordinates {element.box_2d}")
                        else:
                            print(f"  ❌ No matching overlay found for {overlay_num}")
                    else:
                        print(f"  ❌ No overlay number found for {element.description}")
                
                print(f"  Found {len(response.elements)} relevant elements")
                print(f"  {response.elements}")
                return response
            else:
                print("  ❌ No relevant elements detected")
            
            return None
        except Exception as e:
            print(f"❌ Error detecting elements with overlays: {e}")
            return None
    
    def validate_element_coordinates(self, elements: PageElements, page_info: PageInfo) -> PageElements:
        """Validate and clamp all coordinates in the detected elements"""
        try:
            for element in elements.elements:
                if element.box_2d:
                    # box_2d is already validated by the pydantic validator
                    # but we can do additional checks here if needed
                    y_min, x_min, y_max, x_max = element.box_2d
                    
                    # Ensure coordinates are within 0-1000 range (Gemini format)
                    y_min = clamp_coordinate(y_min, 0, 1000)
                    x_min = clamp_coordinate(x_min, 0, 1000)
                    y_max = clamp_coordinate(y_max, 0, 1000)
                    x_max = clamp_coordinate(x_max, 0, 1000)
                    
                    # Ensure min < max
                    if y_min >= y_max:
                        y_max = y_min + 1
                    if x_min >= x_max:
                        x_max = x_min + 1
                        
                    element.box_2d = [y_min, x_min, y_max, x_max]
            
        except Exception as e:
            print(f"⚠️ Error validating element coordinates: {e}")
        
        return elements
