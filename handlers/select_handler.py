"""
Handles select field interactions (both traditional and custom dropdowns).
"""
import time
from typing import Optional

from playwright.sync_api import Page
from models import ActionStep, PageElements, PageInfo, DetectedElement
from utils import SelectorUtils
from vision_utils import validate_and_clamp_coordinates, get_gemini_box_2d_center_pixels


class ShouldUseClickInsteadError(ValueError):
    """
    Special exception raised when select handler determines the element is not a select
    and should be handled with a click action instead.
    
    This allows the action executor to automatically convert select actions to click actions.
    """
    def __init__(self, message: str, option_text: Optional[str] = None):
        super().__init__(message)
        self.option_text = option_text
        self.should_retry_as_click = True


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
                    element.box_2d,
                    page_info.width,
                    page_info.height,
                    page_info.doc_width,
                    page_info.doc_height,
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

        # Scroll into view if the point is outside the current viewport
        try:
            self.page.evaluate(
                "(coords) => { window.scrollTo({ top: Math.max(0, coords.y - 200), left: Math.max(0, coords.x - 200), behavior: 'instant' }); }",
                {"x": x, "y": y},
            )
        except Exception:
            pass

        # Clamp to current viewport for clicking after scroll
        try:
            vp = self.page.viewport_size or {"width": page_info.width, "height": page_info.height}
            x, y = validate_and_clamp_coordinates(x, y, vp.get("width", page_info.width), vp.get("height", page_info.height))
        except Exception:
            x, y = validate_and_clamp_coordinates(x, y, page_info.width, page_info.height)

        # Priority order for selector resolution:
        # 1. Refined selector from executor (most reliable - already validated)
        # 2. element_label from overlay metadata (reliable - comes from overlay detection)
        # 3. Coordinate-based resolution (fallback - may be inaccurate with multiple elements)
        
        selector_hint = getattr(self, "selector_override", None)
        if selector_hint:
            print(f"    Using refined selector from executor: {selector_hint}")
            # Clear it after use
            try:
                delattr(self, "selector_override")
            except Exception:
                pass
        elif element and element.element_label:
            # Prioritize element_label if it looks like a valid CSS selector
            # element_label is set by plan_generator from overlay metadata (id, name, or cssSelector)
            label = element.element_label
            if label and (label.startswith('#') or label.startswith('[') or label.startswith('.')):
                selector_hint = label
                print(f"    Using selector from overlay metadata: {selector_hint}")
        
        # Only fall back to coordinate-based resolution if we don't have a selector yet
        if not selector_hint:
            selector_hint = self.selector_utils.get_element_selector_from_coordinates(x, y)
            if selector_hint:
                print(f"    Using coordinate-based selector resolution: {selector_hint}")
        
        if selector_hint:
            print(f"    Vision selector resolved for '{target_description}': {selector_hint}")

        # Determine if this is a select element from the selector
        tag_from_selector = ""
        role_from_selector = ""
        if selector_hint:
            try:
                tag_from_selector = self.page.evaluate(
                    "(sel) => { const el = document.querySelector(sel); return el ? el.tagName.toLowerCase() : ''; }",
                    selector_hint,
                )
                role_from_selector = self.page.evaluate(
                    "(sel) => { const el = document.querySelector(sel); return el ? (el.getAttribute('role') || '').toLowerCase() : ''; }",
                    selector_hint,
                )
            except Exception:
                tag_from_selector = ""
                role_from_selector = ""

        try:
            # Early validation: Check if this is actually a select element
            # Decide handling path based on selector tag/role first (more reliable than coordinates)
            is_actually_select = False
            is_custom = False
            
            if tag_from_selector == "select":
                is_actually_select = True
                is_custom = False
                print(f"    Element is native <select> (from selector)")
            elif role_from_selector in ("combobox", "listbox"):
                is_actually_select = True
                is_custom = True
                print(f"    Element is custom (role={role_from_selector})")
            else:
                # Fallback: analyze element at coordinates
                html_element_info = self._get_element_at_coordinates(x, y)
                
                if not html_element_info:
                    print("    ⚠️ Could not analyze element at coordinates")
                    # Without element info, we cannot confirm it's a select - default to False
                    is_actually_select = False
                    is_custom = False
                    print(f"    Cannot verify if element is a select - treating as NOT a select for safety")
                else:
                    tag_name = html_element_info.get('tagName', 'unknown')
                    role = html_element_info.get('role', '').lower()
                    
                    # Check if it's actually a select
                    if tag_name == 'select':
                        is_actually_select = True
                        is_custom = False
                        print(f"    Element is native <select> (from analysis)")
                    elif role in ('combobox', 'listbox'):
                        is_actually_select = True
                        is_custom = True
                        print(f"    Element is custom select (role={role})")
                    else:
                        # Use conservative detection to determine if it's a custom select
                        # But don't fail early - let the agent try to handle it as a select
                        is_custom = self._is_custom_select(html_element_info)
                        # If detection suggests it's not a select, still try to handle it
                        # The actual select methods will fail if it doesn't work
                        is_actually_select = True  # Trust the agent's decision to use select
                        print(f"    Element analysis: {tag_name} - Will attempt as select (Custom: {is_custom})")
            
            # No early validation - trust the agent's decision to use select
            # If it's not actually a select, the actual selection methods will fail naturally
            
            # Click the select field to open it
            print(f"    Clicking select field at ({x}, {y})")
            self.page.mouse.click(x, y)
            
            # Wait for options to appear
            print("    Waiting for options to appear...")
            time.sleep(0.5)

            if is_custom:
                print("    Detected custom select, using AI for option selection")
                self._handle_custom_select(element, step, page_info, selector_hint)
            else:
                print("    Detected traditional select, using standard selection")
                self._handle_traditional_select(element, step, page_info, selector_hint, click_point=(x, y))

            # Clear override hint after use
            if hasattr(self, "selector_override"):
                delattr(self, "selector_override")
                
        except Exception as e:
            error_msg = str(e)
            print(f"    ❌ Select handling failed: {error_msg}")
            # Provide guidance for the agent when select methods don't work
            if "not found" in error_msg.lower() or "could not select" in error_msg.lower() or "may not be a true select" in error_msg.lower():
                print(f"    💡 HINT: If select methods don't work, this might not be a select field.")
                print(f"    💡 HINT: Try using 'click: <option text>' instead of 'select: <option text>'")
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
                            element.box_2d,
                            page_info.width,
                            page_info.height,
                            page_info.doc_width,
                            page_info.doc_height,
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
        """
        Deterministically determine if an element is a custom select implementation.
        More conservative: only returns True if there are strong indicators of a select.
        This prevents false positives for regular inputs with lists below them.
        """
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
        
        # Only treat as custom select if we have STRONG indicators
        # This prevents false positives for regular inputs with lists below them
        strong_indicators = [
            role == 'combobox',  # Strong indicator - explicit ARIA role
            role == 'listbox',   # Strong indicator - explicit ARIA role
            (aria_expanded in ['true', 'false'] and aria_haspopup in ['true', 'listbox', 'menu']),  # Both together indicate dropdown
        ]
        
        # If we have strong role indicators, it's likely a custom select
        if any(strong_indicators):
            print(f"    Custom select indicators found: tagName={tag_name}, role={role}")
            return True
        
        # For class-based detection, be more conservative
        # Only treat as select if it has multiple indicators together
        class_indicators = [
            has_dropdown_class,
            'select' in class_name and tag_name != 'select',
            'dropdown' in class_name,
            'picker' in class_name,
        ]
        
        # Require at least 2 class indicators to avoid false positives
        if sum(class_indicators) >= 2:
            print(f"    Custom select indicators found: tagName={tag_name}, class={class_name}")
            return True
        
        # Check for HTML patterns, but be more conservative
        html_content = element_info.get('innerHTML', '').lower()
        outer_html = element_info.get('outerHTML', '').lower()
        
        # Only treat as select if we have data attributes (strong indicator)
        if 'data-value' in outer_html or 'data-option' in outer_html:
            print(f"    Custom select data attributes found in {tag_name} element")
            return True
        
        # Default to NOT a select if no strong indicators
        print(f"    No strong select indicators found for {tag_name}, treating as regular element")
        return False
    
    def _handle_traditional_select(
        self,
        element: Optional[DetectedElement],
        step: ActionStep,
        page_info: PageInfo,
        selector_hint: Optional[str] = None,
        click_point: Optional[tuple] = None,
    ) -> None:
        """Handle traditional HTML select element using Playwright's native selectOption API"""
        try:
            # Get the selector for the select element
            # PRIORITY: First try to find an actual <select> element near the coordinates
            query_selector = selector_hint
            
            # If we have coordinates, prioritize finding a <select> element
            if not query_selector:
                if click_point:
                    query_selector = self._find_select_selector_near_point(click_point[0], click_point[1])
                if not query_selector and element and element.box_2d:
                    cx, cy = get_gemini_box_2d_center_pixels(
                        element.box_2d,
                        page_info.width,
                        page_info.height,
                        page_info.doc_width,
                        page_info.doc_height,
                    )
                    query_selector = self._find_select_selector_near_point(cx, cy)
            
            # Fallback: Use generic selector resolution (but verify it's actually a select)
            if not query_selector and element and element.box_2d:
                cx, cy = get_gemini_box_2d_center_pixels(
                    element.box_2d,
                    page_info.width,
                    page_info.height,
                    page_info.doc_width,
                    page_info.doc_height,
                )
                generic_selector = self.selector_utils.get_element_selector_from_coordinates(cx, cy)
                # Only use generic selector if it's actually a select element
                if generic_selector:
                    try:
                        actual_tag = self.page.evaluate(
                            "(sel) => { const el = document.querySelector(sel); return el ? el.tagName.toLowerCase() : ''; }",
                            generic_selector,
                        )
                        if actual_tag == 'select':
                            query_selector = generic_selector
                            print(f"    ✅ Generic selector resolved to <select>: {query_selector}")
                        else:
                            print(f"    ⚠️ Generic selector '{generic_selector}' is <{actual_tag}>, not <select>. Will try to find select nearby.")
                            # Try to find select near this element
                            query_selector = self._find_select_selector_near_point(cx, cy)
                    except Exception:
                        # If we can't verify, try to find select nearby anyway
                        query_selector = self._find_select_selector_near_point(cx, cy)

            if not query_selector:
                query_selector = 'select'

            # Create locator for the select element (guard invalid selectors)
            try:
                select_locator = self.page.locator(query_selector)
                # force a quick access to ensure selector is valid
                _ = select_locator.first
            except Exception as err:
                print(f"    ⚠️ Invalid selector '{query_selector}', trying nearest select: {err}")
                alt = self._find_select_selector_near_point(click_point[0], click_point[1]) if click_point else None
                if alt:
                    print(f"    ✅ Using fallback select selector: {alt}")
                    select_locator = self.page.locator(alt)
                    query_selector = alt
                else:
                    raise
            
            # Try to verify the element tag, but don't fail early - just log if it's not a select
            try:
                actual_tag = self.page.evaluate(
                    "(sel) => { const el = document.querySelector(sel); return el ? el.tagName.toLowerCase() : ''; }",
                    query_selector,
                )
                if actual_tag != 'select':
                    print(f"    ⚠️ Element with selector '{query_selector}' is a <{actual_tag}>, not a <select>. Will attempt selection anyway.")
            except Exception as e:
                print(f"    ⚠️ Could not verify element tag: {e}. Will attempt selection anyway.")
            
            try:
                select_locator.scroll_into_view_if_needed()
            except Exception:
                pass
            
            # Check if select exists and is visible
            try:
                count = select_locator.count()
                if count == 0:
                    print(f"    ⚠️ No select element found with selector: {query_selector}")
                    raise ValueError(f"Select element not found: {query_selector}")
                elif count > 1:
                    print(f"    ⚠️ Selector '{query_selector}' matches {count} elements. Using first() but this may be ambiguous.")
                    # Try to use a more specific selector if we have coordinates
                    if click_point:
                        better_selector = self._find_select_selector_near_point(click_point[0], click_point[1])
                        if better_selector:
                            print(f"    ✅ Found more specific select selector: {better_selector}")
                            select_locator = self.page.locator(better_selector)
                            query_selector = better_selector
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
                
                # All selectOption methods failed - try fallback: custom select handling
                print(f"    ⚠️ Could not select '{opt}' using any selectOption method")
                print(f"    🔄 Attempting fallback: trying custom select handling route")
                try:
                    # Try custom select handling as fallback
                    # This handles cases where native selectOption doesn't work due to custom JS handlers
                    self._handle_custom_select(element, step, page_info, selector_hint)
                    print(f"    ✅ Successfully selected '{opt}' via custom select fallback")
                    return
                except Exception as custom_err:
                    print(f"    ⚠️ Custom select fallback also failed: {custom_err}")
                    
                    # Last resort: try clicking on the option text in the page
                    print(f"    🔄 Attempting final fallback: clicking on option text '{opt}'")
                    try:
                        option_locator = self.page.get_by_text(opt, exact=False).first
                        if option_locator.count() > 0:
                            option_locator.click()
                            print(f"    ✅ Successfully clicked option '{opt}' via text click fallback")
                            return
                    except Exception as click_err:
                        print(f"    ⚠️ Text click fallback also failed: {click_err}")
                
                # If all methods fail, raise error - let the agent decide what to do next
                error_msg = (
                    f"Could not select '{opt}' using any select methods. "
                    f"Tried by label, value, index, partial match, custom select handling, and text click, but all failed."
                )
                raise ValueError(error_msg)

            # When no option specified, extract available options for agent to choose
            # Get all available options
            available_options = self.page.evaluate("""
                (sel) => {
                    const select = document.querySelector(sel);
                    if (!select) return [];
                    return Array.from(select.options || [])
                        .filter(opt => !opt.disabled)
                        .map(opt => ({
                            value: opt.value || '',
                            text: (opt.textContent || '').trim(),
                            label: opt.label || '',
                            index: opt.index
                        }));
                }
            """, query_selector)
            
            if not available_options:
                print("    ⚠️ No options available in select")
                return
            
            print(f"    🔍 No option specified; {len(available_options)} options available")
            option_texts = [opt['text'] for opt in available_options if opt['text']]
            print(f"    📋 Available options: {', '.join(option_texts[:10])}{'...' if len(option_texts) > 10 else ''}")
            
            # Store available options in handler for agent to access
            # This allows the agent to see what options are available and choose one
            self.available_options = available_options
            self.pending_select_selector = query_selector
            # Get field description from element if available
            field_description = None
            if element:
                field_description = getattr(element, 'description', None) or getattr(element, 'element_label', None) or getattr(element, 'element_type', None)
            self.pending_select_field = field_description or "select field"
            
            # Don't auto-select - let the agent determine which option to choose
            # The action executor will check for this information and store it
            print(f"    💡 Select field opened but no option specified. Agent should determine which option to select.")
            return
            
        except Exception as e:
            error_msg = str(e)
            print(f"    ❌ Traditional select handling failed: {error_msg}")
            # Provide guidance for the agent
            if "may not be a true select" in error_msg.lower():
                print(f"    💡 HINT: This element may not be a select field. Use 'click:' instead of 'select:'")
            raise

    def _find_select_selector_near_point(self, x: int, y: int) -> Optional[str]:
        """
        Try to resolve a unique CSS selector for a select element near given coordinates.
        Prefers id/name, else returns a positional selector like select:nth-of-type(k).
        """
        try:
            result = self.page.evaluate(
                """
                ({x, y}) => {
                    const pointEl = document.elementFromPoint(x, y);
                    const candidates = Array.from(document.querySelectorAll('select'));
                    if (!candidates.length) return null;

                    const distance = (rect) => {
                        const cx = rect.left + rect.width / 2;
                        const cy = rect.top + rect.height / 2;
                        const dx = cx - x;
                        const dy = cy - y;
                        return Math.sqrt(dx*dx + dy*dy);
                    };

                    let best = null;
                    let bestDist = Infinity;

                    candidates.forEach((sel, idx) => {
                        const rect = sel.getBoundingClientRect();
                        // must be on screen
                        if (!rect || rect.width === 0 || rect.height === 0) return;
                        const d = distance(rect);
                        if (d < bestDist) {
                            bestDist = d;
                            best = { el: sel, idx };
                        }
                    });

                    if (!best) return null;
                    const el = best.el;

                    if (el.id) return '#' + el.id;
                    if (el.name) return `[name="${el.name}"]`;

                    // try data-testid
                    const testAttrs = ['data-testid','data-test','data-cy','data-qa'];
                    for (const attr of testAttrs) {
                        const val = el.getAttribute(attr);
                        if (val) return `[${attr}="${val}"]`;
                    }

                    // fallback to positional selector among all selects
                    const nth = best.idx + 1;
                    return `select:nth-of-type(${nth})`;
                }
                """,
                {"x": x, "y": y},
            )
            if result:
                print(f"    ✅ Resolved nearest select selector: {result}")
            else:
                print("    ⚠️ Could not resolve select near point, falling back to generic")
            return result or None
        except Exception as err:
            print(f"    ⚠️ Error resolving select near point ({x}, {y}): {err}")
            return None

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
            # Try to open the dropdown/combobox explicitly if we have a selector
            if selector_hint:
                trigger = self.page.locator(selector_hint)
                try:
                    trigger.scroll_into_view_if_needed()
                except Exception:
                    pass
                try:
                    trigger.click()
                    time.sleep(0.1)
                except Exception:
                    pass
                try:
                    trigger_tag = trigger.evaluate("el => el.tagName.toLowerCase()")
                except Exception:
                    trigger_tag = ""
                if trigger_tag == "input" and option_text:
                    try:
                        trigger.fill(option_text)
                        time.sleep(0.1)
                        try:
                            opt_loc = self.page.get_by_role("option", name=option_text)
                            if opt_loc.count():
                                opt_loc.first.click()
                                return
                        except Exception:
                            pass
                        trigger.press("Enter")
                        return
                    except Exception:
                        pass

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
            # If we have option text, raise error - let the agent decide what to do next
            if option_text:
                error_msg = (
                    f"Could not find option '{option_text}' in custom select. "
                    f"Tried multiple methods but no matching options were found."
                )
                raise ValueError(error_msg)

        except Exception as err:
            error_msg = str(err)
            print(f"    ❌ Custom select handling failed: {error_msg}")
            # Provide guidance for the agent
            if "may not be a true select" not in error_msg.lower():
                if option_text:
                    print(f"    💡 HINT: If custom select methods don't work, try using 'click: {option_text}' instead")
            raise
