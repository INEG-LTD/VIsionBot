"""
Utilities for generating and working with CSS selectors.
"""
import time
from playwright.sync_api import Page


class SelectorUtils:
    """Utilities for CSS selector generation and field value retrieval"""
    
    def __init__(self, page: Page):
        self.page = page
    
    def set_page(self, page: Page) -> None:
        if page and page is not self.page:
            self.page = page
    
    def get_element_selector_from_coordinates(self, x: int, y: int) -> str:
        """Get a CSS selector for the element at the given coordinates
        
        Args:
            x, y: Pixel coordinates - should be viewport/client coordinates (relative to visible viewport)
                 If document coordinates are passed, they will be converted to viewport coordinates
        """
        try:
            # First, remove any overlays that might interfere
            self._remove_overlays()
            time.sleep(0.2)  # Longer delay to ensure overlays are fully removed
            
            try:
                from utils.event_logger import get_event_logger
                get_event_logger().system_debug(f"Looking for element at pixel coordinates: ({x}, {y})")
            except Exception:
                pass
            
            js_code = f"""
            (function() {{
                // Convert to viewport/client coordinates if needed
                // elementFromPoint expects viewport coordinates, not document coordinates
                let cx = {x};
                let cy = {y};
                
                // Check if coordinates are outside viewport bounds (might be document coordinates)
                // If so, convert by subtracting scroll offset
                if (cx < 0 || cx > window.innerWidth || cy < 0 || cy > window.innerHeight) {{
                    cx = {x} - window.scrollX;
                    cy = {y} - window.scrollY;
                }}
                
                // Validate coordinates are within viewport
                if (!Number.isFinite(cx) || !Number.isFinite(cy)) {{
                    console.log('Invalid coordinates (non-finite):', {x}, {y});
                    return null;
                }}
                
                if (cx < 0 || cy < 0 || cx >= window.innerWidth || cy >= window.innerHeight) {{
                    console.log('Coordinates outside viewport bounds:', cx, cy, 'viewport:', window.innerWidth, window.innerHeight);
                    // Still try elementFromPoint - sometimes it works slightly outside bounds
                }}
                
                console.log('Looking for element at viewport coordinates: (' + cx + ', ' + cy + ')');
                let element = document.elementFromPoint(cx, cy);
                
                if (!element) {{
                    console.log('No element found at coordinates - element might be hidden or coordinates invalid');
                    return null;
                }}
                
                console.log('Found element:', element.tagName, element.id, element.className, element.type);
                
                // Walk up the DOM tree to find interactive ancestors (similar to element_analyzer)
                // This handles cases where we hit a non-interactive wrapper element
                const clickableSelector = 'a,button,input,select,textarea,[role="button"],[role="link"],[role="combobox"],[role="listbox"],[role="option"]';
                const clickableAncestor = element.closest ? element.closest(clickableSelector) : null;
                if (clickableAncestor) {{
                    console.log('Found interactive ancestor:', clickableAncestor.tagName, clickableAncestor.id);
                    element = clickableAncestor;
                }}
                
                // If still non-interactive and no ancestor found, look for nearby inputs
                const nonInteractiveTags = ['html', 'body', 'header', 'footer', 'nav', 'section', 'div', 'span', 'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'];
                const style = window.getComputedStyle(element);
                const isNonInteractive = nonInteractiveTags.includes(element.tagName.toLowerCase()) && 
                    !element.onclick && 
                    !element.getAttribute('onclick') &&
                    !element.hasAttribute('tabindex') &&
                    style.pointerEvents !== 'none' &&
                    element.tagName.toLowerCase() !== 'div';
                
                if (isNonInteractive && !clickableAncestor) {{
                    // Look for an input element nearby (within the same area)
                    const inputs = document.querySelectorAll('input, select, textarea, button, a');
                    let closestInput = null;
                    let closestDistance = Infinity;
                    
                    inputs.forEach(input => {{
                        const rect = input.getBoundingClientRect();
                        // Skip hidden elements
                        const inputStyle = window.getComputedStyle(input);
                        if (inputStyle.display === 'none' || inputStyle.visibility === 'hidden') {{
                            return;
                        }}
                        
                        const centerX = rect.left + rect.width / 2;
                        const centerY = rect.top + rect.height / 2;
                        const distance = Math.sqrt(Math.pow(centerX - cx, 2) + Math.pow(centerY - cy, 2));
                        
                        // Increased search radius to 100 pixels for better reliability
                        if (distance < closestDistance && distance < 100) {{
                            closestDistance = distance;
                            closestInput = input;
                        }}
                    }});
                    
                    if (closestInput) {{
                        console.log('Found nearby interactive element:', closestInput.tagName, closestInput.id, closestInput.type, 'distance:', closestDistance.toFixed(1));
                        element = closestInput; // Use the nearby input instead
                    }} else {{
                        console.log('No nearby interactive element found within 100px');
                        // Don't return null yet - still try to generate a selector for the element we found
                    }}
                }}
                
                // Priority 1: ID (most reliable)
                if (element.id) {{
                    // Check if ID contains characters that make it invalid for CSS ID selectors
                    // CSS ID selectors cannot contain: spaces, colons, periods, hashes, brackets, etc.
                    const invalidChars = /[ :.#\[\]]/;
                    if (invalidChars.test(element.id)) {{
                        // Use attribute selector for IDs with invalid characters
                        const attrSelector = '[id="' + element.id.replace(/"/g, '\\"') + '"]';
                        console.log('Using attribute selector for invalid ID:', attrSelector);
                        return attrSelector;
                    }} else {{
                        console.log('Using ID selector:', '#' + element.id);
                        return '#' + element.id;
                    }}
                }}
                
                // Priority 2: Name attribute
                if (element.name) {{
                    console.log('Using name selector:', '[name="' + element.name + '"]');
                    return '[name="' + element.name + '"]';
                }}
                
                // Priority 3: data-testid or similar test attributes
                const testAttrs = ['data-testid', 'data-test', 'data-cy', 'data-qa'];
                for (const attr of testAttrs) {{
                    const attrValue = element.getAttribute(attr);
                    if (attrValue) {{
                        console.log('Using test attribute selector:', '[' + attr + '="' + attrValue + '"]');
                        return '[' + attr + '="' + attrValue + '"]';
                    }}
                }}
                
                // Priority 4: Type + class combination for inputs
                if (element.tagName.toLowerCase() === 'input' && element.type) {{
                    if (element.className) {{
                        const classes = element.className.split(' ').filter(c => c.length > 0);
                        if (classes.length > 0) {{
                            const selector = 'input[type="' + element.type + '"].' + classes[0];
                            console.log('Using input+class selector:', selector);
                            return selector;
                        }}
                    }}
                    const selector = 'input[type="' + element.type + '"]';
                    console.log('Using input type selector:', selector);
                    return selector;
                }}
                
                // Priority 5: Class name (if unique enough)
                if (element.className) {{
                    const classes = element.className.split(' ').filter(c => c.length > 0);
                    if (classes.length > 0) {{
                        // Try first class
                        const testSelector = '.' + classes[0];
                        const matchingElements = document.querySelectorAll(testSelector);
                        if (matchingElements.length === 1) {{
                            console.log('Using unique class selector:', testSelector);
                            return testSelector;
                        }}
                        
                        // Try combination of tag + class
                        const tagClassSelector = element.tagName.toLowerCase() + '.' + classes[0];
                        const tagClassMatches = document.querySelectorAll(tagClassSelector);
                        if (tagClassMatches.length === 1) {{
                            console.log('Using tag+class selector:', tagClassSelector);
                            return tagClassSelector;
                        }}
                    }}
                }}
                
                console.log('No reliable selector found, returning null');
                return null;
            }})();
            """
            
            selector = self.page.evaluate(js_code)
            if selector:
                try:
                    from utils.event_logger import get_event_logger
                    get_event_logger().system_debug(f"Found reliable selector: {selector}")
                except Exception:
                    pass
                return selector
            else:
                print(f"    ⚠️ First attempt failed, trying alternative approach...")
                # Try a simpler approach - just get any selector for the element
                simple_selector = self._get_simple_selector(x, y)
                if simple_selector:
                    print(f"    ✅ Found simple selector: {simple_selector}")
                    return simple_selector
                else:
                    print(f"    ❌ Could not find any selector for element at ({x}, {y})")
                    return ""
            
        except Exception as e:
            print(f"    ⚠️ Error getting element selector: {e}")
            return ""

    def _get_simple_selector(self, x: int, y: int) -> str:
        """Get a simple CSS selector for the element, less strict than the main method"""
        try:
            js_code = f"""
            (function() {{
                // Convert to viewport coordinates (same logic as main method)
                let cx = {x};
                let cy = {y};
                
                if (cx < 0 || cx > window.innerWidth || cy < 0 || cy > window.innerHeight) {{
                    cx = {x} - window.scrollX;
                    cy = {y} - window.scrollY;
                }}
                
                if (!Number.isFinite(cx) || !Number.isFinite(cy)) {{
                    return null;
                }}
                
                const element = document.elementFromPoint(cx, cy);
                if (!element) return null;
                
                // Try to find interactive ancestor
                const clickableSelector = 'a,button,input,select,textarea,[role="button"],[role="link"]';
                const clickableAncestor = element.closest ? element.closest(clickableSelector) : null;
                const targetElement = clickableAncestor || element;
                
                console.log('Simple selector - Found element:', targetElement.tagName, targetElement.id, targetElement.className);
                
                // Just try the most basic selectors without validation
                if (targetElement.id) {{
                    // Check if ID contains characters that make it invalid for CSS ID selectors
                    const invalidChars = /[ :.#\[\]]/;
                    if (invalidChars.test(targetElement.id)) {{
                        return '[id="' + targetElement.id.replace(/"/g, '\\"') + '"]';
                    }} else {{
                        return '#' + targetElement.id;
                    }}
                }}
                
                if (targetElement.name) {{
                    return '[name="' + targetElement.name + '"]';
                }}
                
                // For inputs, just use the type
                if (targetElement.tagName.toLowerCase() === 'input' && targetElement.type) {{
                    return 'input[type="' + targetElement.type + '"]';
                }}
                
                // Try first class if available
                if (targetElement.className) {{
                    const classes = targetElement.className.split(' ').filter(c => c.length > 0);
                    if (classes.length > 0) {{
                        return '.' + classes[0];
                    }}
                }}
                
                return null;
            }})();
            """
            return self.page.evaluate(js_code) or ""
        except Exception as e:
            print(f"    ⚠️ Error in simple selector: {e}")
            return ""
    
    def get_field_value_by_selector(self, selector: str) -> str:
        """Get the current value of a field using its CSS selector"""
        try:
            js_code = f"""
            (function() {{
                const element = document.querySelector('{selector}');
                if (!element) return '';
                return element.value || element.textContent || '';
            }})();
            """
            value = self.page.evaluate(js_code)
            return str(value).strip()
        except Exception as e:
            print(f"    ⚠️ Error getting field value by selector: {e}")
            return ''

    def get_field_value(self, x: int, y: int) -> str:
        """Get the current value of a field at the given coordinates"""
        try:
            js_code = f"""
            (function() {{
                const element = document.elementFromPoint({x}, {y});
                if (!element) return '';
                return element.value || element.textContent || '';
            }})();
            """
            value = self.page.evaluate(js_code)
            return str(value).strip()
        except Exception as e:
            print(f"    ⚠️ Error getting field value: {e}")
            return ''
    
    def _remove_overlays(self) -> None:
        """Remove element overlays that might interfere with selector detection"""
        try:
            js_code = """
            (function() {
                const overlays = document.querySelectorAll('.automation-element-overlay');
                overlays.forEach(overlay => overlay.remove());
                return overlays.length;
            })();
            """
            self.page.evaluate(js_code)
        except Exception:
            pass  # Ignore errors in cleanup
