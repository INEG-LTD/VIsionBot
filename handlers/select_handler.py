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
        print(f"    Debug: overlay_index = {step.overlay_index}")
        print(f"    Debug: elements count = {len(elements.elements)}")
        print(f"    Debug: step coordinates = ({step.x}, {step.y})")
        
        if step.overlay_index is None:
            print("    ❌ No overlay index provided")
            raise ValueError("No overlay index provided for select field")
        
        # Find element by overlay_number instead of array index
        element = None
        for elem in elements.elements:
            if elem.overlay_number == step.overlay_index:
                element = elem
                break
        
        if element is None:
            available_overlays = [str(e.overlay_number) for e in elements.elements if e.overlay_number is not None]
            print(f"    ❌ No element found with overlay number {step.overlay_index}")
            print(f"    Available overlay numbers: {', '.join(available_overlays) if available_overlays else 'none'}")
            raise ValueError(f"No element found with overlay number {step.overlay_index} for select field")
        x, y = self._get_click_coordinates(step, elements, page_info)
        target_description = element.description or element.element_label or element.element_type
        selector_hint: Optional[str] = None

        if x is None or y is None:
            raise ValueError("Could not determine coordinates for select field")

        # Validate and clamp coordinates
        x, y = validate_and_clamp_coordinates(x, y, page_info.width, page_info.height)

        selector_hint = self.selector_utils.get_element_selector_from_coordinates(x, y)
        if selector_hint:
            print(f"    Vision selector resolved for '{target_description}': {selector_hint}")

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
                self._handle_custom_select(element, step, page_info, selector_hint)
            else:
                print("    Detected traditional select, using standard selection")
                self._handle_traditional_select(element, step, page_info, selector_hint)
                
        except Exception as e:
            print(f"    ❌ Select handling failed: {e}")
            raise
    
    def _get_click_coordinates(self, step: ActionStep, elements: PageElements, page_info: PageInfo) -> tuple:
        """Get click coordinates from step or element"""
        if step.x is not None and step.y is not None:
            return int(step.x), int(step.y)
        
        if step.overlay_index is not None:
            # Find element by overlay_number instead of array index
            for element in elements.elements:
                if element.overlay_number == step.overlay_index:
                    if element.box_2d:
                        center_x, center_y = get_gemini_box_2d_center_pixels(
                            element.box_2d, page_info.width, page_info.height
                        )
                        if center_x > 0 or center_y > 0:
                            return center_x, center_y
                    break
        
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
    
    def _handle_traditional_select(
        self,
        element: DetectedElement,
        step: ActionStep,
        page_info: PageInfo,
        selector_hint: Optional[str] = None,
    ) -> None:
        """Handle traditional HTML select element"""
        try:
            query_selector = selector_hint
            if not query_selector and element.box_2d:
                cx, cy = get_gemini_box_2d_center_pixels(element.box_2d, page_info.width, page_info.height)
                query_selector = self.selector_utils.get_element_selector_from_coordinates(cx, cy)

            if not query_selector:
                query_selector = 'select'

            if step.select_option_text:
                opt = step.select_option_text
                
                # Debug: List all available options
                available_options = self.page.evaluate("""
                    (sel) => {
                        const select = document.querySelector(sel);
                        if (!select) return [];
                        return Array.from(select.options || []).map(opt => ({
                            value: opt.value,
                            text: opt.textContent?.trim(),
                            label: opt.label?.trim()
                        }));
                    }
                """, query_selector)
                print(f"    🔍 Available options: {available_options}")
                
                try:
                    self.page.select_option(query_selector, label=opt)
                    print(f"    ✅ Selected '{opt}' via selector {query_selector}")
                    return
                except Exception:
                    print(f"    ⚠️ select_option by label failed for '{opt}', trying text matching")
                try:
                    self.page.select_option(query_selector, value=opt)
                    print(f"    ✅ Selected value '{opt}' via selector {query_selector}")
                    return
                except Exception:
                    print(f"    ⚠️ select_option by value failed for '{opt}'")
                
                # Try partial matching
                try:
                    for option in available_options:
                        if option['text'] and opt.lower() in option['text'].lower():
                            self.page.select_option(query_selector, value=option['value'])
                            print(f"    ✅ Selected '{option['text']}' (partial match for '{opt}')")
                            return
                except Exception as e:
                    print(f"    ⚠️ Partial matching failed: {e}")

            # Fallback: pick first non-disabled option different from placeholder
            selected = self.page.evaluate("""
                (sel) => {
                    const select = document.querySelector(sel);
                    if (!select) return null;
                    const options = Array.from(select.options || []);
                    for (let i = 0; i < options.length; i++) {
                        const opt = options[i];
                        if (opt.disabled) continue;
                        if (i === 0 && opt.value === '') continue;
                        select.value = opt.value;
                        select.dispatchEvent(new Event('input', { bubbles: true }));
                        select.dispatchEvent(new Event('change', { bubbles: true }));
                        return { value: opt.value, text: opt.textContent };
                    }
                    return null;
                }
            """, query_selector)

            if selected:
                print(f"    ✅ Selected option '{selected['text']}' via DOM fallback")
            else:
                print("    ⚠️ No suitable option found to select")
            
        except Exception as e:
            print(f"    ❌ Traditional select handling failed: {e}")

    def _handle_custom_select(
        self,
        element: DetectedElement,
        step: ActionStep,
        page_info: PageInfo,
        selector_hint: Optional[str] = None,
    ) -> None:
        """Handle custom select element with AI assistance"""
        option_text = (step.select_option_text or "").strip()
        try:
            candidate_locators = []
            if option_text:
                if selector_hint:
                    candidate_locators.append(self.page.locator(selector_hint).locator('[role="option"]', has_text=option_text))
                    candidate_locators.append(self.page.locator(selector_hint).locator('text=' + option_text))
                candidate_locators.append(self.page.get_by_role('option', name=option_text))
                candidate_locators.append(self.page.locator('[role="option"]', has_text=option_text))
                candidate_locators.append(self.page.locator('li', has_text=option_text))
                candidate_locators.append(self.page.locator('button', has_text=option_text))
                candidate_locators.append(self.page.locator('div', has_text=option_text))

                for loc in candidate_locators:
                    try:
                        count = loc.count()
                    except Exception:
                        count = 0
                    if not count:
                        continue
                    try:
                        loc.first.click()
                        print(f"    ✅ Clicked custom option '{option_text}'")
                        return
                    except Exception:
                        continue

                print(f"    ⚠️ Could not find option '{option_text}' via candidate locators; attempting fallback")

            fallback_loc = self.page.locator('[role="option"]')
            try:
                fallback_count = fallback_loc.count()
            except Exception:
                fallback_count = 0
            if fallback_count:
                fallback_loc.first.click()
                print("    ✅ Selected first available custom option (fallback)")
                return

            print("    ⚠️ No custom options available after fallback attempts")

        except Exception as err:
            print(f"    ❌ Custom select handling failed: {err}")
