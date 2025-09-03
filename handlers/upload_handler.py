"""
Handles file upload interactions (both traditional and custom implementations).
"""
import time
from typing import Optional

from playwright.sync_api import Page
from models import ActionStep, PageElements, PageInfo, DetectedElement
from utils import SelectorUtils
from vision_utils import validate_and_clamp_coordinates, get_gemini_box_2d_center_pixels


class UploadHandler:
    """Handles file upload interactions"""
    
    def __init__(self, page: Page):
        self.page = page
        self.selector_utils = SelectorUtils(page)
    
    def handle_upload_field(self, step: ActionStep, elements: PageElements, page_info: PageInfo) -> None:
        """Execute a specialized upload field interaction"""
        print("  Handling upload field")
        
        # Debug information
        print(f"    Debug: target_element_index = {step.target_element_index}")
        print(f"    Debug: elements count = {len(elements.elements)}")
        print(f"    Debug: step coordinates = ({step.x}, {step.y})")
        
        if step.target_element_index is None or step.target_element_index >= len(elements.elements):
            print(f"    ❌ Invalid target element index: {step.target_element_index} (max: {len(elements.elements) - 1})")
            raise ValueError(f"Invalid target element index {step.target_element_index} for upload field (elements count: {len(elements.elements)})")
        
        element = elements.elements[step.target_element_index]
        x, y = self._get_click_coordinates(step, elements, page_info)
        
        if x is None or y is None:
            raise ValueError("Could not determine coordinates for upload field")
        
        # Validate and clamp coordinates
        x, y = validate_and_clamp_coordinates(x, y, page_info.width, page_info.height)
        
        try:
            # Step 1: Analyze the HTML element at these coordinates to determine type
            html_element_info = self._get_element_at_coordinates(x, y)
            
            if not html_element_info:
                print("    ⚠️ Could not analyze element, falling back to traditional upload handling")
                is_custom = False
                html_element_info = {}
            else:
                is_custom = self._is_custom_upload(html_element_info)
                tag_name = html_element_info.get('tagName', 'unknown')
                print(f"    Element analysis: {tag_name} - Custom: {is_custom}")
            
            # Step 2: Handle based on upload type
            if is_custom:
                print("    Detected custom upload, looking for hidden file input")
                success = self._handle_custom_upload(element, step, page_info, html_element_info)
            else:
                print("    Detected traditional file input")
                success = self._handle_traditional_upload(element, step, page_info, html_element_info)
            
            if not success:
                print("    ⚠️ Upload handling may not have completed successfully")
                
        except Exception as e:
            print(f"    ❌ Upload handling failed: {e}")
            raise
    
    def _get_click_coordinates(self, step: ActionStep, elements: PageElements, page_info: PageInfo) -> tuple:
        """Get click coordinates from step or element"""
        if step.x is not None and step.y is not None:
            return int(step.x), int(step.y)
        
        if step.target_element_index is not None:
            if 0 <= step.target_element_index < len(elements.elements):
                element = elements.elements[step.target_element_index]
                if element.box_2d:
                    center_x, center_y = get_gemini_box_2d_center_pixels(
                        element.box_2d, page_info.width, page_info.height
                    )
                    if center_x > 0 or center_y > 0:
                        return center_x, center_y
        
        return None, None
    
    def _get_element_at_coordinates(self, x: int, y: int) -> dict:
        """Get HTML element information at the specified coordinates"""
        # Placeholder - would need the full implementation
        return {}
    
    def _is_custom_upload(self, element_info: dict) -> bool:
        """Determine if an element is a custom upload implementation"""
        # Placeholder - would need the full implementation
        return False
    
    def _handle_traditional_upload(self, element: DetectedElement, step: ActionStep, page_info: PageInfo, html_element_info: dict) -> bool:
        """Handle traditional file input upload"""
        # Placeholder - would need the full implementation
        print("    Traditional upload handling not yet implemented in modular version")
        return False
    
    def _handle_custom_upload(self, element: DetectedElement, step: ActionStep, page_info: PageInfo, html_element_info: dict) -> bool:
        """Handle custom upload implementation"""
        # Placeholder - would need the full implementation
        print("    Custom upload handling not yet implemented in modular version")
        return False
