"""
Utilities for generating and working with CSS selectors.
"""
import time
from playwright.sync_api import Page


class SelectorUtils:
    """Utilities for CSS selector generation and field value retrieval"""
    
    def __init__(self, page: Page):
        self.page = page
    
    def get_element_selector_from_coordinates(self, x: int, y: int) -> str:
        """Get a CSS selector for the element at the given coordinates"""
        try:
            # First, remove any overlays that might interfere
            self._remove_overlays()
            time.sleep(0.2)  # Longer delay to ensure overlays are fully removed
            
            print(f"    🔍 Looking for element at pixel coordinates: ({x}, {y})")
            
            js_code = f"""
            (function() {{
                console.log('Looking for element at coordinates: {x}, {y}');
                const element = document.elementFromPoint({x}, {y});
                
                if (!element) {{
                    console.log('No element found at coordinates');
                    return null;
                }}
                
                console.log('Found element:', element.tagName, element.id, element.className, element.type);
                
                // Skip non-interactive elements early
                const nonInteractiveTags = ['html', 'body', 'header', 'footer', 'nav', 'section', 'div', 'span', 'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'];
                if (nonInteractiveTags.includes(element.tagName.toLowerCase()) && 
                    !element.onclick && 
                    !element.getAttribute('onclick') &&
                    !element.hasAttribute('tabindex') &&
                    element.tagName.toLowerCase() !== 'div') {{
                    
                    // Look for an input element nearby (within the same area)
                    const inputs = document.querySelectorAll('input, select, textarea');
                    let closestInput = null;
                    let closestDistance = Infinity;
                    
                    inputs.forEach(input => {{
                        const rect = input.getBoundingClientRect();
                        const centerX = rect.left + rect.width / 2;
                        const centerY = rect.top + rect.height / 2;
                        const distance = Math.sqrt(Math.pow(centerX - {x}, 2) + Math.pow(centerY - {y}, 2));
                        
                        if (distance < closestDistance && distance < 50) {{ // Within 50 pixels
                            closestDistance = distance;
                            closestInput = input;
                        }}
                    }});
                    
                    if (closestInput) {{
                        console.log('Found nearby input element:', closestInput.tagName, closestInput.id, closestInput.type);
                        element = closestInput; // Use the nearby input instead
                    }} else {{
                        console.log('No nearby input found, element might not be interactive');
                        return null;
                    }}
                }}
                
                // Priority 1: ID (most reliable)
                if (element.id) {{
                    console.log('Using ID selector:', '#' + element.id);
                    return '#' + element.id;
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
                print(f"    ✅ Found reliable selector: {selector}")
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
                const element = document.elementFromPoint({x}, {y});
                if (!element) return null;
                
                console.log('Simple selector - Found element:', element.tagName, element.id, element.className);
                
                // Just try the most basic selectors without validation
                if (element.id) {{
                    return '#' + element.id;
                }}
                
                if (element.name) {{
                    return '[name="' + element.name + '"]';
                }}
                
                // For inputs, just use the type
                if (element.tagName.toLowerCase() === 'input' && element.type) {{
                    return 'input[type="' + element.type + '"]';
                }}
                
                // Try first class if available
                if (element.className) {{
                    const classes = element.className.split(' ').filter(c => c.length > 0);
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
