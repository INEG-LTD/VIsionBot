"""
Element analysis utilities.
"""
from typing import Any, Dict

from playwright.sync_api import Page


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

