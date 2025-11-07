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
    
    def set_page(self, page: Page) -> None:
        if not page or page is self.page:
            return
        self.page = page
        if hasattr(self.selector_utils, "set_page"):
            self.selector_utils.set_page(page)
        else:
            self.selector_utils.page = page

    def handle_select_field(self, step: ActionStep, elements: PageElements, page_info: PageInfo) -> None:
        """Execute a specialized select field interaction"""
        print("  Handling select field")
        
        # Debug information
        print(f"    Debug: overlay_index = {step.overlay_index}")
        print(f"    Debug: elements count = {len(elements.elements)}")
        print(f"    Debug: step coordinates = ({step.x}, {step.y})")
        
        # Get coordinates - prioritize step coordinates, then try overlay_index
        x, y = self._get_click_coordinates(step, elements, page_info)
        
        if x is None or y is None:
            # If we don't have coordinates and no overlay_index, we can't proceed
            if step.overlay_index is None:
                print("    ❌ No overlay index or coordinates provided")
                raise ValueError("No overlay index or coordinates provided for select field")
            
            # Try to find element by overlay_index to get coordinates
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
            
            # Try to get coordinates from element box
            if element.box_2d:
                x, y = get_gemini_box_2d_center_pixels(
                    element.box_2d, page_info.width, page_info.height
                )
        
        if x is None or y is None:
            raise ValueError("Could not determine coordinates for select field")
        
        # Get element info for description (if available)
        element = None
        if step.overlay_index is not None:
            for elem in elements.elements:
                if elem.overlay_number == step.overlay_index:
                    element = elem
                    break
        
        target_description = (
            element.description if element else None
        ) or (
            element.element_label if element else None
        ) or (
            element.element_type if element else None
        ) or "select field"
        
        selector_hint: Optional[str] = None

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
        element: Optional[DetectedElement],
        step: ActionStep,
        page_info: PageInfo,
        selector_hint: Optional[str] = None,
    ) -> None:
        """Handle traditional HTML select element using Playwright's native selectOption API"""
        try:
            # Get the selector for the select element
            query_selector = selector_hint
            if not query_selector and element and element.box_2d:
                cx, cy = get_gemini_box_2d_center_pixels(element.box_2d, page_info.width, page_info.height)
                query_selector = self.selector_utils.get_element_selector_from_coordinates(cx, cy)

            if not query_selector:
                query_selector = 'select'

            # Create locator for the select element
            select_locator = self.page.locator(query_selector)
            
            # Check if select exists and is visible
            try:
                count = select_locator.count()
                if count == 0:
                    print(f"    ⚠️ No select element found with selector: {query_selector}")
                    raise ValueError(f"Select element not found: {query_selector}")
            except Exception as e:
                print(f"    ⚠️ Error locating select element: {e}")
                raise

            # If we have a specific option to select
            if step.select_option_text:
                opt = step.select_option_text.strip()
                
                # Debug: List all available options
                available_options = self.page.evaluate("""
                    (sel) => {
                        const select = document.querySelector(sel);
                        if (!select) return [];
                        return Array.from(select.options || []).map(opt => ({
                            value: opt.value,
                            text: opt.textContent?.trim(),
                            label: opt.label?.trim(),
                            index: opt.index
                        }));
                    }
                """, query_selector)
                print(f"    🔍 Available options: {available_options}")
                
                # Try Playwright's native selectOption API
                # 1. Try by label (recommended - most common)
                try:
                    select_locator.select_option(label=opt)
                    print(f"    ✅ Selected '{opt}' by label via selectOption")
                    return
                except Exception as e:
                    print(f"    ⚠️ selectOption by label failed for '{opt}': {e}")
                
                # 2. Try by value
                try:
                    select_locator.select_option(value=opt)
                    print(f"    ✅ Selected '{opt}' by value via selectOption")
                    return
                except Exception as e:
                    print(f"    ⚠️ selectOption by value failed for '{opt}': {e}")
                
                # 3. Try by index (if opt is a number)
                try:
                    index = int(opt)
                    select_locator.select_option(index=index)
                    print(f"    ✅ Selected option at index {index} via selectOption")
                    return
                except (ValueError, Exception):
                    pass
                
                # 4. Try partial matching on text/label
                for option in available_options:
                    option_text = option.get('text') or option.get('label') or ''
                    option_value = option.get('value') or ''
                    
                    if opt.lower() in option_text.lower() or opt.lower() == option_value.lower():
                        try:
                            select_locator.select_option(value=option_value)
                            print(f"    ✅ Selected '{option_text}' (partial match for '{opt}') via selectOption")
                            return
                        except Exception as e:
                            print(f"    ⚠️ Failed to select matched option: {e}")
                            continue
                
                print(f"    ⚠️ Could not select '{opt}' using any method")
                raise ValueError(f"Option '{opt}' not found in select element")

            # Fallback: pick first non-disabled option (if no specific option provided)
            # Use Playwright's native API for this too
            try:
                # Get first available option
                first_option = self.page.evaluate("""
                    (sel) => {
                        const select = document.querySelector(sel);
                        if (!select) return null;
                        const options = Array.from(select.options || []);
                        for (let i = 0; i < options.length; i++) {
                            const opt = options[i];
                            if (opt.disabled) continue;
                            if (i === 0 && opt.value === '' && opt.textContent.trim() === '') continue;
                            return { value: opt.value, text: opt.textContent?.trim(), index: i };
                        }
                        return null;
                    }
                """, query_selector)
                
                if first_option:
                    try:
                        # Try to select by value first
                        select_locator.select_option(value=first_option['value'])
                        print(f"    ✅ Selected option '{first_option['text']}' via selectOption fallback")
                        return
                    except Exception:
                        # Fallback to index
                        select_locator.select_option(index=first_option['index'])
                        print(f"    ✅ Selected option '{first_option['text']}' by index via selectOption fallback")
                        return
                else:
                    print("    ⚠️ No suitable option found to select")
            
            except Exception as e:
                print(f"    ⚠️ Fallback selection failed: {e}")
                # Final fallback: use evaluate to set value directly
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
            raise

    def _handle_custom_select(
        self,
        element: Optional[DetectedElement],
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
