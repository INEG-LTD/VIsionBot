"""
Handles select field interactions (both traditional and custom dropdowns).
"""
import time
from typing import Optional

from playwright.sync_api import Page
from models import ActionStep, PageElements, PageInfo, DetectedElement
from utils import SelectorUtils
from vision_utils import validate_and_clamp_coordinates, get_gemini_box_2d_center_pixels


class SelectHandler:
    """Handles select field interactions"""
    
    def __init__(self, page: Page):
        self.page = page
        self.selector_utils = SelectorUtils(page)
    
    def handle_select_field(self, step: ActionStep, elements: PageElements, page_info: PageInfo) -> None:
        """Execute a specialized select field interaction"""
        print("  Handling select field")
        
        # Debug information
        print(f"    Debug: target_element_index = {step.target_element_index}")
        print(f"    Debug: elements count = {len(elements.elements)}")
        print(f"    Debug: step coordinates = ({step.x}, {step.y})")
        
        if step.target_element_index is None or step.target_element_index >= len(elements.elements):
            print(f"    ❌ Invalid target element index: {step.target_element_index} (max: {len(elements.elements) - 1})")
            raise ValueError(f"Invalid target element index {step.target_element_index} for select field (elements count: {len(elements.elements)})")
        
        element = elements.elements[step.target_element_index]
        x, y = self._get_click_coordinates(step, elements, page_info)
        
        if x is None or y is None:
            raise ValueError("Could not determine coordinates for select field")
        
        # Validate and clamp coordinates
        x, y = validate_and_clamp_coordinates(x, y, page_info.width, page_info.height)
        
        try:
            # Step 1: Analyze the HTML element at these coordinates to determine type
            html_element_info = self._get_element_at_coordinates(x, y)
            
            if not html_element_info:
                print("    ⚠️ Could not analyze element, falling back to traditional select handling")
                is_custom = False
                html_element_info = {}
            else:
                is_custom = self._is_custom_select(html_element_info)
                tag_name = html_element_info.get('tagName', 'unknown')
                print(f"    Element analysis: {tag_name} - Custom: {is_custom}")
            
            # Step 2: Click the select field to open it
            print(f"    Clicking select field at ({x}, {y})")
            self.page.mouse.click(x, y)
            
            # Step 3: Wait for options to appear
            print("    Waiting for options to appear...")
            time.sleep(0.5)
            
            # Step 4: Handle based on determined type
            if is_custom:
                print("    Detected custom select, using AI for option selection")
                self._handle_custom_select(element, step, page_info)
            else:
                print("    Detected traditional select, using standard selection")
                self._handle_traditional_select(element, step, page_info)
                
        except Exception as e:
            print(f"    ❌ Select handling failed: {e}")
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
        try:
            element_info = self.page.evaluate("""
                (coords) => {
                    const x = coords.x;
                    const y = coords.y;
                    
                    if (!Number.isFinite(x) || !Number.isFinite(y)) {
                        return null;
                    }
                    
                    const element = document.elementFromPoint(x, y);
                    if (!element) return null;
                    
                    return {
                        tagName: element.tagName.toLowerCase(),
                        id: element.id || '',
                        className: element.className || '',
                        role: element.getAttribute('role') || '',
                        type: element.type || '',
                        ariaExpanded: element.getAttribute('aria-expanded') || '',
                        ariaHaspopup: element.getAttribute('aria-haspopup') || '',
                        innerHTML: element.innerHTML.substring(0, 200),
                        outerHTML: element.outerHTML.substring(0, 500),
                        hasSelectChild: element.querySelector('select') !== null,
                        hasInputChild: element.querySelector('input') !== null,
                        hasDropdownClass: element.className.toLowerCase().includes('dropdown') || 
                                        element.className.toLowerCase().includes('select') ||
                                        element.className.toLowerCase().includes('picker'),
                        parentTagName: element.parentElement ? element.parentElement.tagName.toLowerCase() : '',
                        parentClassName: element.parentElement ? element.parentElement.className || '' : '',
                        isContentEditable: element.contentEditable === 'true',
                        hasClickHandler: element.onclick !== null || element.addEventListener !== undefined
                    };
                }
            """, {"x": x, "y": y})
            
            return element_info or {}
        except Exception as e:
            print(f"    ⚠️ Error getting element at coordinates ({x}, {y}): {e}")
            return {}
    
    def _is_custom_select(self, element_info: dict) -> bool:
        """Deterministically determine if an element is a custom select implementation"""
        if not element_info:
            return False
        
        tag_name = element_info.get('tagName', '').lower()
        role = element_info.get('role', '').lower()
        class_name = element_info.get('className', '').lower()
        aria_expanded = element_info.get('ariaExpanded', '').lower()
        aria_haspopup = element_info.get('ariaHaspopup', '').lower()
        has_dropdown_class = element_info.get('hasDropdownClass', False)
        
        # Definitely traditional select
        if tag_name == 'select':
            return False
        
        # Strong indicators of custom select
        custom_indicators = [
            role == 'combobox',
            role == 'listbox', 
            aria_expanded in ['true', 'false'],
            aria_haspopup in ['true', 'listbox', 'menu'],
            has_dropdown_class,
            'select' in class_name and tag_name != 'select',
            'dropdown' in class_name,
            'picker' in class_name,
            'combobox' in class_name
        ]
        
        # If any strong indicator is present, it's likely custom
        if any(custom_indicators):
            print(f"    Custom select indicators found: tagName={tag_name}, role={role}, class={class_name}")
            return True
        
        # Check for common custom select patterns in HTML structure
        html_content = element_info.get('innerHTML', '').lower()
        outer_html = element_info.get('outerHTML', '').lower()
        
        html_indicators = [
            'option' in html_content and tag_name != 'select',
            'dropdown' in html_content,
            'menu' in html_content and tag_name == 'div',
            'data-value' in outer_html,
            'data-option' in outer_html
        ]
        
        if any(html_indicators):
            print(f"    Custom select HTML patterns found in {tag_name} element")
            return True
        
        # Default to traditional if no custom indicators
        print("    No custom indicators found, treating as traditional")
        return False
    
    def _handle_traditional_select(self, element: DetectedElement, step: ActionStep, page_info: PageInfo):
        """Handle traditional HTML select element"""
        try:
            # Try to find the select element and get its options
            select_options = self.page.evaluate("""
                () => {
                    const selects = document.querySelectorAll('select');
                    for (let select of selects) {
                        const rect = select.getBoundingClientRect();
                        if (rect.width > 0 && rect.height > 0) {
                            return Array.from(select.options).map((opt, index) => ({
                                value: opt.value,
                                text: opt.textContent.trim(),
                                index: index,
                                disabled: opt.disabled
                            }));
                        }
                    }
                    return [];
                }
            """)
            
            if not select_options or len(select_options) <= 1:
                print("    ⚠️ No valid options found in traditional select")
                return
            
            # Select the first non-disabled, non-placeholder option
            for option in select_options[1:]:  # Skip first option (usually placeholder)
                if not option['disabled']:
                    print(f"    Selecting option: {option['text']}")
                    
                    # Use Playwright's select_option method
                    select_element = self.page.query_selector('select')
                    if select_element:
                        select_element.select_option(value=option['value'])
                        print(f"    ✅ Selected: {option['text']}")
                        return
            
            print("    ⚠️ No suitable option found to select")
            
        except Exception as e:
            print(f"    ❌ Traditional select handling failed: {e}")
    
    def _handle_custom_select(self, element: DetectedElement, step: ActionStep, page_info: PageInfo):
        """Handle custom select element with AI assistance"""
        # This would need the full custom select logic from the original file
        # For now, just a placeholder
        print("    Custom select handling not yet implemented in modular version")
        pass
