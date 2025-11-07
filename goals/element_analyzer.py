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
        self._description_cache = {}  # Cache for AI-generated descriptions
    
    def set_page(self, page: Page) -> None:
        """Update the underlying Playwright page reference."""
        if not page or page is self.page:
            return
        self.page = page
        self._description_cache.clear()
    
    def analyze_element_at_coordinates(self, x: int, y: int) -> dict:
        """
        Returns:
        {
            tagName, elementType, text, innerText, attributes, isClickable,
            bounds: { x, y, width, height },
            style: { display, visibility, opacity, cursor },
            parentContext, xpath
        }
        """
        try:
            return self.page.evaluate(
                """({ x, y }) => {
                    // Check if coordinates are already client coordinates (viewport-relative)
                    let cx, cy;
                    if (x >= 0 && x <= window.innerWidth && y >= 0 && y <= window.innerHeight) {
                        // Coordinates are already client coordinates (from viewport-relative detection)
                        cx = x;
                        cy = y;
                    } else {
                        // Convert page coordinates to client coordinates
                        cx = x - window.scrollX;
                        cy = y - window.scrollY;
                    }

                    if (!Number.isFinite(cx) || !Number.isFinite(cy)) return null;
                    if (cx < 0 || cy < 0 || cx >= window.innerWidth || cy >= window.innerHeight) return null;

                    let element = document.elementFromPoint(cx, cy);
                    if (!element) return null;

                    // Heuristic: prefer a clickable ancestor (anchors/buttons/etc.) for evaluation
                    const clickableSelector = 'a,button,[role="link"],[role="button"],input,select,textarea';
                    const clickableAncestor = element.closest ? element.closest(clickableSelector) : null;
                    if (clickableAncestor) {
                        element = clickableAncestor;
                    }

                    const rect = element.getBoundingClientRect();
                    const style = getComputedStyle(element);

                    // Texts
                    const text = (element.textContent || '').trim();
                    const innerText = element.innerText ? element.innerText.trim() : '';

                    // Attributes
                    const attributes = {};
                    for (const attr of element.attributes) {
                        attributes[attr.name] = attr.value;
                    }

                    // Parent context (trimmed & shortened)
                    const parent = element.parentElement;
                    const parentText = parent ? (parent.textContent || '').trim().slice(0, 200) : '';

                    // Element typing
                    const tagName = element.tagName.toLowerCase();
                    const role = element.getAttribute('role') || '';
                    const type = element.getAttribute('type') || '';

                    let elementType = tagName;
                    if (tagName === 'input') {
                        elementType = `${tagName}[${type || 'text'}]`;
                    } else if (role) {
                        elementType = `${tagName}[role=${role}]`;
                    }

                    // Use browser's native clickability detection
                    const isClickable = (
                        element.clickable || 
                        element.tagName === 'BUTTON' || 
                        element.tagName === 'A' ||
                        (element.tagName === 'INPUT' && ['button','submit','checkbox','radio'].includes(type)) ||
                        role === 'button' || 
                        role === 'link' ||
                        style.cursor === 'pointer' ||
                        (element.hasAttribute('tabindex') && element.getAttribute('tabindex') !== '-1')
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
                }""",
                {"x": x, "y": y},  # pass coordinates
            ) or {}
        except Exception as e:
            print(f"[ElementAnalyzer] Error in analyze_element_at_coordinates: {e}")
            return {"error": "Failed to analyze element"}

    def analyze_element_by_selector(self, selector: str) -> dict:
        """Analyze a specific element identified by a CSS selector.

        Returns same shape as analyze_element_at_coordinates.
        """
        try:
            return self.page.evaluate(
                """(sel) => {
                    const element = document.querySelector(sel);
                    if (!element) return null;
                    const rect = element.getBoundingClientRect();
                    const style = getComputedStyle(element);
                    const tagName = element.tagName.toLowerCase();
                    const role = element.getAttribute('role') || '';
                    const type = element.getAttribute('type') || '';
                    const text = (element.textContent || '').trim();
                    const innerText = element.innerText ? element.innerText.trim() : '';
                    const attributes = {};
                    for (const attr of element.attributes) {
                        attributes[attr.name] = attr.value;
                    }
                    // Use browser's native clickability detection
                    const isClickable = (
                        element.clickable || 
                        element.tagName === 'BUTTON' || 
                        element.tagName === 'A' ||
                        (element.tagName === 'INPUT' && ['button','submit','checkbox','radio'].includes(type)) ||
                        role === 'button' || 
                        role === 'link' ||
                        style.cursor === 'pointer' ||
                        (element.hasAttribute('tabindex') && element.getAttribute('tabindex') !== '-1')
                    );
                    return {
                        tagName,
                        elementType: role ? `${tagName}[role=${role}]` : (tagName === 'input' ? `${tagName}[${type || 'text'}]` : tagName),
                        text,
                        innerText,
                        attributes,
                        isClickable,
                        bounds: { x: rect.x, y: rect.y, width: rect.width, height: rect.height },
                        style: { display: style.display, visibility: style.visibility, opacity: style.opacity, cursor: style.cursor },
                        parentContext: element.parentElement ? (element.parentElement.textContent || '').trim().slice(0,200) : '',
                        xpath: null
                    };
                }""",
                selector,
            ) or {}
        except Exception as e:
            print(f"[ElementAnalyzer] Error in analyze_element_by_selector: {e}")
            return {"error": "Failed to analyze element by selector"}

    
    def get_element_description_with_ai(self, element_info: Dict[str, Any], screenshot: bytes, x: int, y: int) -> str:
        """
        Use AI to generate a natural description of the element.
        This helps with matching user intent to actual elements.
        Uses caching to avoid duplicate AI calls for the same element.
        """
        try:
            # Create cache key based on element properties and coordinates
            cache_key = f"{x}_{y}_{hash(str(element_info.get('elementType', '')))}_{hash(str(element_info.get('text', ''))[:50])}"
            
            # Check cache first
            if cache_key in self._description_cache:
                print(f"[ElementAnalyzer] Using cached description for element at ({x}, {y})")
                return self._description_cache[cache_key]
            
            # Create a focused prompt about the specific element
            element_summary = f"""
            Element at coordinates ({x}, {y}):
            - Type: {element_info.get('elementType', 'unknown')}
            - Text: {element_info.get('text', '')[:100]}
            - Clickable: {element_info.get('isClickable', False)}
            - Attributes: {str(element_info.get('attributes', {}))[:200]}
            """
            
            print(f"[ElementAnalyzer] Element summary: {element_summary}")
            
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
                # image=screenshot
            )
            description_str = str(description).strip()
            
            # Cache the result
            self._description_cache[cache_key] = description_str
            print(f"[GoalFramework] Element description with AI: {description_str}")
            
            return description_str
            
        except Exception as e:
            print(f"[ElementAnalyzer] Error in get_element_description_with_ai: {e}")
            # Fallback to basic description
            element_type = element_info.get('elementType', 'element')
            text = element_info.get('text', '')[:50]
            fallback_desc = f"{element_type} containing '{text}'" if text else element_type
            print(f"[GoalFramework] Element description with fallback: {fallback_desc}")
            return fallback_desc
