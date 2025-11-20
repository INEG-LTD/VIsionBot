"""
Handles datetime field interactions with smart format detection.
"""
import re
# from typing import Optional  # Not currently used

from playwright.sync_api import Page
from models import ActionStep, PageElements, PageInfo  # DetectedElement not used
from utils import SelectorUtils
from vision_utils import validate_and_clamp_coordinates
from session_tracker import InteractionType


class DateTimeHandler:
    """Handles datetime field interactions"""
    
    def __init__(self, page: Page, session_tracker=None):
        self.page = page
        self.selector_utils = SelectorUtils(page)
        self.session_tracker = session_tracker
    
    def set_page(self, page: Page) -> None:
        if not page or page is self.page:
            return
        self.page = page
        if hasattr(self.selector_utils, "set_page"):
            self.selector_utils.set_page(page)
        else:
            self.selector_utils.page = page

    def handle_datetime_field(self, step: ActionStep, elements: PageElements, page_info: PageInfo) -> None:
        """Execute datetime field interaction using direct fill() with coordinates"""
        print("  Handling datetime field with direct fill()")
        
        # Debug information
        print(f"    Debug: overlay_index = {step.overlay_index}")
        print(f"    Debug: elements count = {len(elements.elements)}")
        print(f"    Debug: step coordinates = ({step.x}, {step.y})")
        
        element_selector = None
        detected_element = None
        
        # Find element by overlay_number instead of array index
        if step.overlay_index is not None:
            for elem in elements.elements:
                if elem.overlay_number == step.overlay_index:
                    detected_element = elem
                    break

        # Get coordinates from the action step or element
        if step.x is not None and step.y is not None:
            x, y = int(step.x), int(step.y)
            print(f"    Using coordinates from step: ({x}, {y})")
        elif step.overlay_index is not None and detected_element is not None:
            # Get coordinates from element
            from vision_utils import get_gemini_box_2d_center_pixels
            element = detected_element
            if element.box_2d:
                x, y = get_gemini_box_2d_center_pixels(
                    element.box_2d, page_info.width, page_info.height
                )
                print(f"    Using coordinates from element {step.overlay_index}: ({x}, {y})")
            else:
                raise ValueError("Could not determine coordinates for datetime field")
        else:
            available_overlays = [str(e.overlay_number) for e in elements.elements if e.overlay_number is not None]
            print(f"    ❌ No element found with overlay number {step.overlay_index}")
            print(f"    Available overlay numbers: {', '.join(available_overlays) if available_overlays else 'none'}")
            raise ValueError(f"No element found with overlay number {step.overlay_index} for datetime field")
        
        # Validate and clamp coordinates
        x, y = validate_and_clamp_coordinates(x, y, page_info.width, page_info.height)
        
        try:
            # Get the target date we want to set (use provided value or default)
            target_date = step.datetime_value or "2024-12-15"
            
            # Get element selector from coordinates
            if not element_selector:
                element_selector = self.selector_utils.get_element_selector_from_coordinates(x, y)
            if not element_selector:
                raise ValueError("Could not determine element selector from coordinates")

            try:
                center = self.page.evaluate(
                    "(sel) => { const el = document.querySelector(sel); if(!el) return null; const r = el.getBoundingClientRect(); return {x: Math.round(r.left + r.width/2), y: Math.round(r.top + r.height/2)}; }",
                    element_selector,
                )
                if isinstance(center, dict) and 'x' in center and 'y' in center:
                    x, y = int(center['x']), int(center['y'])
                    print(f"    Refined coordinates from selector: ({x}, {y})")
            except Exception:
                pass
            
            print(f"    Found element selector: {element_selector}")
            
            # Detect the input type and format the value appropriately
            formatted_value = self._format_datetime_value(target_date, element_selector)
            print(f"    Filling with formatted value: {formatted_value}")
            
            # Try to fill the field directly
            success = False
            try:
                self.page.fill(element_selector, formatted_value)
                print(f"    ✅ Successfully filled datetime field with: '{formatted_value}'")
                success = True
            except Exception as e:
                print(f"    Fill failed, using fallback click+type: {e}")
                # Fallback to click and type
                self.page.mouse.click(x, y)
                import time
                time.sleep(0.2)
                self.page.keyboard.press("Control+a")
                self.page.keyboard.type(formatted_value, delay=50)
                success = True  # Assume success for fallback
            
            # Press Tab to confirm and move focus
            self.page.keyboard.press("Tab")
            import time
            time.sleep(0.2)
            
            # Verify the final value
            final_value = self.selector_utils.get_field_value_by_selector(element_selector)
            print(f"    ✅ Datetime field updated. Final value: '{final_value}'")
            
            # Record the interaction with goal monitor
            if self.session_tracker:
                self.session_tracker.record_interaction(
                    InteractionType.TYPE,  # Treat datetime as a type interaction
                    coordinates=(x, y),
                    text_input=formatted_value,
                    success=success,
                    error_message=None
                )
            
        except Exception as e:
            print(f"    ❌ Datetime handling failed: {e}")
            raise

    def _format_datetime_value(self, target_value: str, element_selector: str) -> str:
        """Format the datetime value based on the input element type"""
        try:
            # Get the input type from the element
            js_code = f"""
            (function() {{
                const element = document.querySelector('{element_selector}');
                if (!element) return null;
                
                return {{
                    type: element.type || '',
                    tagName: element.tagName.toLowerCase(),
                    step: element.step || '',
                    min: element.min || '',
                    max: element.max || ''
                }};
            }})();
            """
            
            element_info = self.page.evaluate(js_code)
            if not element_info:
                print("    ⚠️ Could not get element info, using original value")
                return target_value
            
            input_type = element_info.get('type', '').lower()
            print(f"    📋 Element type: {input_type}")
            
            # Extract time components from target value
            # Handle different input formats
            if 'T' in target_value:  # ISO format like "2025-09-01T14:30" or "2024-12-15T14:30"
                time_match = re.search(r'T(\d{1,2}):(\d{1,2})', target_value)
            elif ' ' in target_value:  # Format like "2025-09-01 14:30"
                time_match = re.search(r'\s(\d{1,2}):(\d{1,2})', target_value)
            else:  # Just time like "14:30" or date like "2024-12-15"
                time_match = re.search(r'^(\d{1,2}):(\d{1,2})$', target_value)
                if not time_match:
                    # If it's a date, extract default time
                    date_match = re.search(r'(\d{4})-(\d{1,2})-(\d{1,2})', target_value)
                    if date_match:
                        year, month, day = date_match.groups()
                        if input_type == 'date':
                            return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                        elif input_type == 'month':
                            return f"{year}-{month.zfill(2)}"
                        else:
                            # Default time for other types
                            time_match = None
                            hour, minute = "14", "30"  # Default 2:30 PM
                    else:
                        return target_value  # Return as-is if can't parse
            
            if time_match:
                hour, minute = time_match.groups()
            else:
                hour, minute = "14", "30"  # Default 2:30 PM
            
            # Format based on input type
            if input_type == 'time':
                return f"{hour.zfill(2)}:{minute.zfill(2)}"
            elif input_type == 'datetime-local':
                # Extract date part or use default
                date_match = re.search(r'(\d{4})-(\d{1,2})-(\d{1,2})', target_value)
                if date_match:
                    year, month, day = date_match.groups()
                else:
                    year, month, day = "2024", "12", "15"
                return f"{year}-{month.zfill(2)}-{day.zfill(2)}T{hour.zfill(2)}:{minute.zfill(2)}"
            elif input_type == 'date':
                # Extract date part or use default
                date_match = re.search(r'(\d{4})-(\d{1,2})-(\d{1,2})', target_value)
                if date_match:
                    year, month, day = date_match.groups()
                    return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                else:
                    return "2024-12-15"
            elif input_type == 'month':
                # Extract year-month or use default
                date_match = re.search(r'(\d{4})-(\d{1,2})', target_value)
                if date_match:
                    year, month = date_match.groups()
                    return f"{year}-{month.zfill(2)}"
                else:
                    return "2024-12"
            else:
                # For unknown types, try to return the most appropriate format
                if ':' in target_value:
                    return f"{hour.zfill(2)}:{minute.zfill(2)}"
                else:
                    return target_value
                    
        except Exception as e:
            print(f"    ⚠️ Error formatting datetime value: {e}")
            return target_value
