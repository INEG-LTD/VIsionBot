"""
Manages numbered overlays for element detection.
"""
from typing import List, Dict, Any
import time

from playwright.sync_api import Page

from models import PageInfo


class OverlayManager:
    """Manages numbered overlays on interactive elements"""
    
    def __init__(self, page: Page):
        self.page = page
    
    def create_numbered_overlays(self, page_info: PageInfo) -> List[Dict[str, Any]]:
        """Draw numbered overlays on interactive elements and return their normalized coordinates"""
        js_code = f"""
        (function() {{
            // Remove any existing overlays
            const existingOverlays = document.querySelectorAll('.automation-element-overlay');
            existingOverlays.forEach(overlay => overlay.remove());
            
            const viewportWidth = {page_info.width};
            const viewportHeight = {page_info.height};
            
            // Function to create a numbered overlay
            function createNumberedOverlay(element, index) {{
                const rect = element.getBoundingClientRect();
                // Tag the element so we can re-locate it later by overlay index
                try {{ element.setAttribute('data-automation-overlay-index', String(index)); }} catch (e) {{}}
                
                // Create main border
                const border = document.createElement('div');
                border.className = 'automation-element-overlay';
                border.style.cssText = `
                    position: fixed;
                    left: ${{rect.left}}px;
                    top: ${{rect.top}}px;
                    width: ${{rect.width}}px;
                    height: ${{rect.height}}px;
                    border: 2px solid #ff0000;
                    pointer-events: none;
                    z-index: 999999;
                    background: rgba(255, 0, 0, 0.1);
                    box-sizing: border-box;
                `;
                
                // Create number label
                const label = document.createElement('div');
                label.className = 'automation-element-overlay';
                label.textContent = index.toString();
                label.style.cssText = `
                    position: fixed;
                    left: ${{rect.left - 2}}px;
                    top: ${{rect.top - 25}}px;
                    background: #ff0000;
                    color: white;
                    padding: 2px 6px;
                    font-size: 12px;
                    font-weight: bold;
                    border-radius: 3px;
                    pointer-events: none;
                    z-index: 1000000;
                    font-family: Arial, sans-serif;
                `;
                
                document.body.appendChild(border);
                document.body.appendChild(label);
                
                // Return element info with normalized coordinates
                return {{
                    index: index,
                    element: element,
                    rect: rect,
                    normalizedCoords: [
                        Math.round((rect.top / viewportHeight) * 1000),     // y_min
                        Math.round((rect.left / viewportWidth) * 1000),     // x_min  
                        Math.round((rect.bottom / viewportHeight) * 1000),  // y_max
                        Math.round((rect.right / viewportWidth) * 1000)     // x_max
                    ],
                    description: element.tagName.toLowerCase() + 
                                (element.textContent ? ': ' + element.textContent.trim().substring(0, 50) : '') +
                                (element.placeholder ? ' [' + element.placeholder + ']' : '') +
                                (element.type ? ' (type=' + element.type + ')' : ''),
                    tagName: element.tagName.toLowerCase(),
                    type: element.type || '',
                    textContent: element.textContent?.trim().substring(0, 100) || '',
                    className: element.className || '',
                    id: element.id || '',
                    placeholder: element.placeholder || '',
                    role: element.getAttribute('role') || '',
                    name: element.getAttribute('name') || '',
                    ariaLabel: element.getAttribute('aria-label') || '',
                    href: element.getAttribute('href') || ''
                }};
            }}
            
            // Function to check if element is likely interactive
            function isLikelyInteractive(element) {{
                const interactiveTags = ['button', 'input', 'select', 'textarea', 'a'];
                if (interactiveTags.includes(element.tagName.toLowerCase())) {{
                    return true;
                }}
                
                const role = element.getAttribute('role');
                if (role && ['button', 'link', 'tab', 'menuitem', 'option', 'combobox', 'listbox', 'textbox'].includes(role)) {{
                    return true;
                }}
                
                if (element.onclick || element.getAttribute('onclick')) {{
                    return true;
                }}
                
                if (element.hasAttribute('tabindex') && element.getAttribute('tabindex') !== '-1') {{
                    return true;
                }}
                
                const className = (element.className || '').toString().toLowerCase();
                const interactiveClasses = [
                    'btn', 'button', 'clickable', 'click', 'dropdown', 'select', 'option',
                    'menu', 'nav', 'link', 'toggle', 'tab', 'accordion', 'modal', 'popup',
                    'form-control', 'input', 'field', 'submit', 'cancel', 'close', 'edit',
                    'delete', 'add', 'remove', 'save', 'search', 'filter', 'sort'
                ];
                if (interactiveClasses.some(cls => className.includes(cls))) {{
                    return true;
                }}
                
                const dataAttrs = Array.from(element.attributes)
                    .filter(attr => attr.name.startsWith('data-'))
                    .map(attr => attr.name.toLowerCase());
                const interactiveDataAttrs = [
                    'data-testid', 'data-test', 'data-cy', 'data-qa', 'data-action',
                    'data-click', 'data-toggle', 'data-target', 'data-dismiss',
                    'data-value', 'data-option', 'data-select'
                ];
                if (dataAttrs.some(attr => interactiveDataAttrs.includes(attr))) {{
                    return true;
                }}
                
                if (element.hasAttribute('aria-label') && element.getAttribute('aria-label').trim()) {{
                    return true;
                }}
                
                const computedStyle = window.getComputedStyle(element);
                if (computedStyle.cursor === 'pointer') {{
                    return true;
                }}
                
                const text = element.textContent?.trim().toLowerCase() || '';
                const interactiveText = [
                    'click', 'submit', 'send', 'save', 'cancel', 'close', 'edit', 'delete',
                    'add', 'remove', 'select', 'choose', 'pick', 'next', 'previous',
                    'continue', 'back', 'login', 'register', 'sign', 'upload', 'download'
                ];
                if (text && interactiveText.some(word => text.includes(word))) {{
                    return true;
                }}
                
                return false;
            }}
            
            const allElements = document.querySelectorAll('*');
            const elementData = [];
            let index = 1;
            
            allElements.forEach(element => {{
                if (!isLikelyInteractive(element)) {{
                    return;
                }}
                
                const style = window.getComputedStyle(element);
                if (style.display === 'none' || style.visibility === 'hidden' || 
                    element.offsetWidth === 0 || element.offsetHeight === 0) {{
                    return;
                }}
                
                const rect = element.getBoundingClientRect();
                // Skip if too small
                if (rect.width < 10 || rect.height < 10) {{
                    return;
                }}
                // Skip if element is completely outside the viewport (no intersection)
                if (rect.bottom <= 0 || rect.right <= 0 || rect.top >= viewportHeight || rect.left >= viewportWidth) {{
                    return;
                }}
                
                elementData.push(createNumberedOverlay(element, index));
                index++;
            }});
            
            console.log(`Created ${{elementData.length}} numbered overlays`);
            return elementData;
        }})();
        """
        
        try:
            element_data = self.page.evaluate(js_code)
            print(f"🔢 Created {len(element_data)} numbered element overlays")
            return element_data or []
        except Exception as e:
            print(f"⚠️ Error creating numbered overlays: {e}")
            return []
    
    def remove_overlays(self) -> None:
        """Remove all element overlays"""
        js_code = """
        (function() {
            const overlays = document.querySelectorAll('.automation-element-overlay');
            const count = overlays.length;
            overlays.forEach(overlay => overlay.remove());
            console.log(`Removed ${count} element overlays`);
            return count;
        })();
        """
        
        try:
            count = self.page.evaluate(js_code)
            if count > 0:
                print(f"🧹 Removed {count} element overlays")
        except Exception as e:
            print(f"⚠️ Error removing element overlays: {e}")
