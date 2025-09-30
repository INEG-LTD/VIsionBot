"""
Detects UI elements using AI vision models with numbered overlays.
"""
from typing import List, Dict, Any, Optional

from ai_utils import generate_model
from models import PageElements, PageInfo


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
        page_info: PageInfo,
        target_context_guard: Optional[str] = None,
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
        {f"- Required surrounding context for the correct element: {target_context_guard}" if target_context_guard else ""}

        ## Available Elements
        The screenshot shows numbered red overlays (1, 2, 3, etc.) on interactive elements:
        {chr(10).join(element_summary)}

        ## Instructions
        1. Look at the screenshot and identify which numbered elements are most relevant to achieving the user's goal
        2. Focus on elements that directly help accomplish the goal
        {"3. Ignore any overlay whose surrounding content conflicts with the required context." if target_context_guard else ''}
        {"4. Return information about the relevant elements using their exact overlay numbers" if target_context_guard else "3. Return information about the relevant elements using their exact overlay numbers"}
        {"5. Return at least 3 candidates for the user to choose from" if target_context_guard else "4. Return at least 3 candidates for the user to choose from"}

        ## Output Schema
        Return a JSON list where each object represents a relevant element with:
        - "element_label": The label of the element
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
