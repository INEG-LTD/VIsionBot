"""
Handles file upload interactions (both traditional and custom implementations).
"""
import time
from typing import Optional

from playwright.sync_api import Page
from models import ActionStep, PageElements, PageInfo, DetectedElement
from utils import SelectorUtils
from vision_utils import validate_and_clamp_coordinates, get_gemini_box_2d_center_pixels


class UploadHandler:
    """Handles file upload interactions"""
    
    def __init__(self, page: Page):
        self.page = page
        self.selector_utils = SelectorUtils(page)
    
    def handle_upload_field(self, step: ActionStep, elements: PageElements, page_info: PageInfo) -> None:
        """Execute a specialized upload field interaction"""
        print("  Handling upload field")
        
        # Debug information
        print(f"    Debug: overlay_index = {step.overlay_index}")
        print(f"    Debug: elements count = {len(elements.elements)}")
        print(f"    Debug: step coordinates = ({step.x}, {step.y})")
        
        if step.overlay_index is None or step.overlay_index >= len(elements.elements):
            print(f"    ❌ Invalid target element index: {step.overlay_index} (max: {len(elements.elements) - 1})")
            raise ValueError(f"Invalid target element index {step.overlay_index} for upload field (elements count: {len(elements.elements)})")
        
        element = elements.elements[step.overlay_index]
        target_description = element.description or element.element_label or element.element_type
        if not step.upload_file_path:
            raise ValueError("Upload action is missing a file path to upload")

        x, y = self._get_click_coordinates(step, elements, page_info)
        if x is not None and y is not None:
            x, y = validate_and_clamp_coordinates(x, y, page_info.width, page_info.height)
        else:
            raise ValueError("Could not determine upload field coordinates for selector resolution")

        selector = self.selector_utils.get_element_selector_from_coordinates(x, y)
        if selector:
            print(f"    Vision selector resolved for '{target_description}': {selector}")
        else:
            raise ValueError("Could not resolve upload field selector")

        try:
            self.page.set_input_files(selector, step.upload_file_path)
            print(f"    ✅ Set file '{step.upload_file_path}' on upload field")
            return
        except Exception as err:
            print(f"    ⚠️ Upload via selector failed ({err}), searching for hidden file input")
            fallback_selector = self._locate_hidden_file_input(selector, x, y)
            if fallback_selector:
                try:
                    self.page.set_input_files(fallback_selector, step.upload_file_path)
                    print(f"    ✅ Set file '{step.upload_file_path}' using fallback selector {fallback_selector}")
                    return
                except Exception as inner_err:
                    print(f"    ❌ Fallback upload also failed: {inner_err}")
            raise
    
    def _get_click_coordinates(self, step: ActionStep, elements: PageElements, page_info: PageInfo) -> tuple:
        """Get click coordinates from step or element"""
        if step.x is not None and step.y is not None:
            return int(step.x), int(step.y)
        
        if step.overlay_index is not None:
            if 0 <= step.overlay_index < len(elements.elements):
                element = elements.elements[step.overlay_index]
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
            return self.page.evaluate(
                """
                ({ x, y }) => {
                    const el = document.elementFromPoint(x, y);
                    if (!el) return null;
                    const rect = el.getBoundingClientRect();
                    const attrs = {};
                    for (const attr of el.attributes) {
                        attrs[attr.name] = attr.value;
                    }
                    return {
                        tagName: el.tagName.toLowerCase(),
                        type: el.getAttribute('type') || '',
                        id: el.id || '',
                        className: el.className || '',
                        role: el.getAttribute('role') || '',
                        labelFor: el.getAttribute('for') || '',
                        hasFileInputChild: !!el.querySelector('input[type="file"]'),
                        attributes: attrs,
                        rect: {
                            left: rect.left,
                            top: rect.top,
                            width: rect.width,
                            height: rect.height,
                        },
                    };
                }
                """,
                {"x": x, "y": y},
            ) or {}
        except Exception as err:
            print(f"    ⚠️ Error inspecting upload element at ({x}, {y}): {err}")
            return {}

    def _locate_hidden_file_input(self, selector: Optional[str], x: Optional[int], y: Optional[int]) -> Optional[str]:
        """Search for a hidden file input related to the target element."""
        token = f"auto-upload-{int(time.time() * 1000)}"
        try:
            fallback_selector = self.page.evaluate(
                """
                ({ selector, point, token }) => {
                    const mark = (element) => {
                        if (!element) return null;
                        try { element.setAttribute('data-automation-upload', token); } catch (e) {}
                        return `[data-automation-upload="${token}"]`;
                    };

                    const queue = [];
                    const visited = new Set();

                    if (selector) {
                        const base = document.querySelector(selector);
                        if (base) queue.push(base);
                    }
                    if (point && Number.isFinite(point.x) && Number.isFinite(point.y)) {
                        const hit = document.elementFromPoint(point.x, point.y);
                        if (hit && !queue.includes(hit)) queue.push(hit);
                    }

                    const inspect = (node) => {
                        if (!node) return null;
                        if (node.tagName && node.tagName.toLowerCase() === 'input' && (node.type || '').toLowerCase() === 'file') {
                            return mark(node);
                        }
                        const labelFor = node.getAttribute && node.getAttribute('for');
                        if (labelFor) {
                            const target = document.getElementById(labelFor);
                            if (target && target.type && target.type.toLowerCase() === 'file') {
                                return mark(target);
                            }
                        }
                        return null;
                    };

                    while (queue.length) {
                        const current = queue.shift();
                        if (!current || visited.has(current)) continue;
                        visited.add(current);
                        const found = inspect(current);
                        if (found) return found;
                        if (current.querySelectorAll) {
                            const candidates = current.querySelectorAll('input[type="file"], label, button, div, span');
                            for (const cand of candidates) {
                                const hit = inspect(cand);
                                if (hit) return hit;
                                queue.push(cand);
                            }
                        }
                        if (current.parentElement && !visited.has(current.parentElement)) {
                            queue.push(current.parentElement);
                        }
                    }

                    const globalFileInput = document.querySelector('input[type="file"]');
                    return mark(globalFileInput);
                }
                """,
                {
                    "selector": selector,
                    "point": {"x": x, "y": y} if x is not None and y is not None else None,
                    "token": token,
                },
            )
            return fallback_selector
        except Exception as err:
            print(f"    ⚠️ Failed to locate hidden file input: {err}")
            return None
