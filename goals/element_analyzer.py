"""
Element analysis utilities for the goal framework.
"""
from typing import Any, Dict

from playwright.sync_api import Page

from ai_utils import generate_text


class ElementAnalyzer:
    """Helper class for analyzing elements at specific coordinates"""
    
    def __init__(self, page: Page):
        self.page = page
    
    def analyze_element_at_coordinates(self, x: int, y: int) -> Dict[str, Any]:
        """
        Analyze what element exists at the given coordinates.
        
        Returns comprehensive information about the target element.
        """
        try:
            element_info = self.page.evaluate("""
            (coords) => {
                const x = coords.x;
                const y = coords.y;
                const element = document.elementFromPoint(x, y);
                if (!element) return null;
                
                const rect = element.getBoundingClientRect();
                const style = getComputedStyle(element);
                
                // Get text content
                const text = element.textContent?.trim() || '';
                const innerText = element.innerText?.trim() || '';
                
                // Get attributes
                const attributes = {};
                for (const attr of element.attributes) {
                    attributes[attr.name] = attr.value;
                }
                
                // Get parent context
                const parent = element.parentElement;
                const parentText = parent ? parent.textContent?.trim().slice(0, 200) : '';
                
                // Determine element type
                const tagName = element.tagName.toLowerCase();
                const role = element.getAttribute('role') || '';
                const type = element.getAttribute('type') || '';
                
                let elementType = tagName;
                if (tagName === 'input') {
                    elementType = `${tagName}[${type || 'text'}]`;
                } else if (role) {
                    elementType = `${tagName}[role=${role}]`;
                }
                
                // Check if clickable
                const isClickable = (
                    tagName === 'button' ||
                    tagName === 'a' ||
                    (tagName === 'input' && ['button', 'submit'].includes(type)) ||
                    role === 'button' ||
                    style.cursor === 'pointer' ||
                    element.onclick !== null
                );
                
                return {
                    tagName,
                    elementType,
                    text,
                    innerText,
                    attributes,
                    isClickable,
                    bounds: {
                        x: rect.x,
                        y: rect.y,
                        width: rect.width,
                        height: rect.height
                    },
                    style: {
                        display: style.display,
                        visibility: style.visibility,
                        opacity: style.opacity,
                        cursor: style.cursor
                    },
                    parentContext: parentText,
                    xpath: null // We'll calculate this if needed
                };
            }
            """, {"x": x, "y": y})
            
            return element_info or {}
            
        except Exception as e:
            print(f"[ElementAnalyzer] Error in analyze_element_at_coordinates: {e}")
            return {"error": "Failed to analyze element"}
    
    def get_element_description_with_ai(self, element_info: Dict[str, Any], screenshot: bytes, x: int, y: int) -> str:
        """
        Use AI to generate a natural description of the element.
        This helps with matching user intent to actual elements.
        """
        try:
            # Create a focused prompt about the specific element
            element_summary = f"""
            Element at coordinates ({x}, {y}):
            - Type: {element_info.get('elementType', 'unknown')}
            - Text: {element_info.get('text', '')[:100]}
            - Clickable: {element_info.get('isClickable', False)}
            - Attributes: {str(element_info.get('attributes', {}))[:200]}
            """
            
            system_prompt = f"""
            You are analyzing a UI element to provide a natural description.
            
            Element details:
            {element_summary}
            
            Provide a concise, natural description of this element as a user would refer to it.
            Examples: "Submit button", "Email input field", "First link in navigation", "Login form"
            
            Focus on what the element IS and what it DOES, not technical details.
            """
            
            description = generate_text(
                prompt="Describe this UI element naturally and concisely.",
                system_prompt=system_prompt,
                model="gemini-2.5-flash-lite",
                image=screenshot
            )
            print(f"[GoalFramework] Element description with AI: {description}")
            
            return str(description).strip()
            
        except Exception as e:
            print(f"[ElementAnalyzer] Error in get_element_description_with_ai: {e}")
            # Fallback to basic description
            element_type = element_info.get('elementType', 'element')
            text = element_info.get('text', '')[:50]
            print(f"[GoalFramework] Element description with fallback: {element_type} containing '{text}'")
            if text:
                return f"{element_type} containing '{text}'"
            else:
                print(f"[GoalFramework] Element description with fallback: {element_type}")
                return element_type
