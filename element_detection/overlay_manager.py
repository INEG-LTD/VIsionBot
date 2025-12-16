"""
Manages numbered overlays for element detection.
"""
from typing import List, Dict, Any

from playwright.sync_api import Page

from models import PageInfo


class OverlayManager:
    """Manages numbered overlays on page elements.

    Default behavior overlays likely-interactive elements. You can switch to
    non-discriminatory mode (all visible elements) via the `mode` parameter.
    """
    
    def __init__(self, page: Page):
        self.page = page
    
    def create_numbered_overlays(self, page_info: PageInfo, mode: str = "interactive") -> List[Dict[str, Any]]:
        """Draw numbered overlays and return their normalized coordinates.

        Args:
            page_info: Basic viewport/page sizing info
            mode: 'interactive' (default) or 'all' to include every visible element
        """
        js_code = f"""
        (function() {{
            // Remove any existing overlays
            const existingOverlays = document.querySelectorAll('.automation-element-overlay');
            existingOverlays.forEach(overlay => overlay.remove());
            
            const viewportWidth = {page_info.width};
            const viewportHeight = {page_info.height};
            const MODE = "{mode}";
            const INCLUDE_ALL = (MODE === 'all' || MODE === 'visible');
            
            function collectContextText(element) {{
                const pieces = [];
                let current = element;
                let depth = 0;
                const MAX_DEPTH = 4;
                while (current && current !== document.body && depth < MAX_DEPTH) {{
                    const text = (current.innerText || '').trim();
                    if (text) {{
                        pieces.push(text.replace(/\\s+/g, ' ').substring(0, 180));
                    }}
                    current = current.parentElement;
                    depth += 1;
                }}
                return pieces.join(' | ').substring(0, 360);
            }}

            function collectDataAttributes(element) {{
                const attrs = {{}};
                Array.from(element.attributes || []).forEach(attr => {{
                    if (attr.name && attr.name.startsWith('data-')) {{
                        attrs[attr.name] = attr.value || '';
                    }}
                }});
                return attrs;
            }}

            // Find the associated label/question text for form elements (inputs, radios, checkboxes, selects)
            // This helps distinguish between similar options like "Yes" for different questions
            function findAssociatedLabel(element) {{
                const tagName = element.tagName.toLowerCase();
                const inputTypes = ['input', 'select', 'textarea'];
                
                // Check if element is a form element or contains one (like label containing input)
                let inputElement = element;
                if (!inputTypes.includes(tagName)) {{
                    // Check if element contains an input
                    const nestedInput = element.querySelector('input, select, textarea');
                    if (nestedInput) {{
                        inputElement = nestedInput;
                    }} else {{
                        // Not a form element and doesn't contain one, skip
                        return '';
                    }}
                }}
                
                // 1. Check aria-labelledby attribute (on the input element)
                const labelledBy = inputElement.getAttribute('aria-labelledby');
                if (labelledBy) {{
                    const labelEl = document.getElementById(labelledBy);
                    if (labelEl) {{
                        const labelText = (labelEl.textContent || labelEl.innerText || '').trim();
                        if (labelText) {{
                            return labelText.substring(0, 150);
                        }}
                    }}
                }}
                
                // 2. Check aria-label attribute (on the input element)
                const ariaLabel = inputElement.getAttribute('aria-label');
                if (ariaLabel && ariaLabel.trim()) {{
                    return ariaLabel.trim().substring(0, 150);
                }}
                
                // 3. Check for wrapping label element
                let parent = inputElement.parentElement;
                while (parent && parent !== document.body && parent.tagName.toLowerCase() !== 'form') {{
                    if (parent.tagName.toLowerCase() === 'label') {{
                        const labelText = (parent.textContent || parent.innerText || '').trim();
                        const elemText = (inputElement.textContent || inputElement.value || '').trim();
                        let questionText = labelText;
                        if (elemText && labelText.includes(elemText)) {{
                            questionText = labelText.replace(elemText, '').trim();
                        }}
                        if (questionText) {{
                            return questionText.substring(0, 150);
                        }}
                        break;
                    }}
                    parent = parent.parentElement;
                }}
                
                // 4. Check for label element with 'for' attribute pointing to the input element's id
                if (inputElement.id) {{
                    const labelFor = document.querySelector('label[for="' + inputElement.id + '"]');
                    if (labelFor) {{
                        const labelText = (labelFor.textContent || labelFor.innerText || '').trim();
                        if (labelText) {{
                            return labelText.substring(0, 150);
                        }}
                    }}
                }}
                
                // 5. Check for fieldset legend
                parent = inputElement.parentElement;
                while (parent && parent !== document.body) {{
                    if (parent.tagName.toLowerCase() === 'fieldset') {{
                        const legend = parent.querySelector('legend');
                        if (legend) {{
                            const legendText = (legend.textContent || legend.innerText || '').trim();
                            if (legendText) {{
                                return legendText.substring(0, 150);
                            }}
                        }}
                        break;
                    }}
                    parent = parent.parentElement;
                }}
                
                // 6. For radio/checkbox in lists, find question text from parent container
                // Start from the original element (which might be a label/li) to find the question
                const inputType = inputElement.type || '';
                if (inputType === 'radio' || inputType === 'checkbox') {{
                    // Look up the DOM tree starting from the original element to find the question
                    parent = element.parentElement;
                    let depth = 0;
                    while (parent && parent !== document.body && depth < 4) {{
                        // Look for siblings before the parent that contain the question
                        let sibling = parent.previousElementSibling;
                        let siblingDepth = 0;
                        while (sibling && siblingDepth < 3) {{
                            const siblingText = (sibling.textContent || sibling.innerText || '').trim();
                            // Skip if sibling contains common option text like "Yes", "No", "Other"
                            const commonOptions = ['yes', 'no', 'other', 'ok', 'cancel'];
                            const lowerText = siblingText.toLowerCase();
                            const isOption = commonOptions.some(opt => lowerText === opt || lowerText.startsWith(opt + ' ') || lowerText === opt);
                            
                            // If sibling has question-like text (long, contains question words, or has ?)
                            if (siblingText && !isOption && (siblingText.length > 10 || siblingText.includes('?'))) {{
                                // Check if it looks like a question
                                const questionWords = ['are', 'is', 'do', 'does', 'will', 'can', 'would', 'when', 'what', 'where', 'who', 'how', 'which'];
                                const hasQuestionWord = questionWords.some(word => siblingText.toLowerCase().includes(word));
                                if (hasQuestionWord || siblingText.includes('?') || siblingText.length > 20) {{
                                    // Clean up the text (remove special characters at the end)
                                    const cleaned = siblingText.replace(/[✱*]+/g, '').trim();
                                    return cleaned.substring(0, 150);
                                }}
                            }}
                            sibling = sibling.previousElementSibling;
                            siblingDepth++;
                        }}
                        
                        // Check parent's text content - look for question text before the options
                        const parentText = (parent.textContent || parent.innerText || '').trim();
                        const elemText = (element.textContent || element.value || '').trim();
                        if (parentText && elemText && parentText !== elemText) {{
                            // Try to extract just the question part (before common options)
                            const parentWithoutOptions = parentText.replace(/(Yes|No|Other|YesNoOther)/gi, '').replace(/[✱*]+/g, '').trim();
                            if (parentWithoutOptions.length > 10) {{
                                const questionWords = ['are', 'is', 'do', 'does', 'will', 'can', 'would', 'when', 'what', 'where', 'who', 'how', 'which'];
                                const hasQuestionWord = questionWords.some(word => parentWithoutOptions.toLowerCase().includes(' ' + word + ' '));
                                if (hasQuestionWord || parentWithoutOptions.includes('?') || parentWithoutOptions.length > 20) {{
                                    // Split by newlines or take first sentence
                                    const parts = parentWithoutOptions.split(/[\\n\\r]/);
                                    const firstPart = parts[0].trim();
                                    if (firstPart.length > 10) {{
                                        return firstPart.substring(0, 150);
                                    }}
                                    return parentWithoutOptions.substring(0, 150);
                                }}
                            }}
                        }}
                        
                        parent = parent.parentElement;
                        depth++;
                    }}
                }}
                
                return '';
            }}

            // Function to create a numbered overlay
            function createNumberedOverlay(element, index, coords) {{
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
                
                // Find associated label for form elements
                const fieldLabel = findAssociatedLabel(element);
                
                // For select elements, include available options in description
                // Also detect custom select elements (combobox, listbox) even if they don't have options visible yet
                let selectOptions = '';
                const role = element.getAttribute('role') || '';
                const isNativeSelect = element.tagName.toLowerCase() === 'select';
                const isCustomSelect = role === 'combobox' || role === 'listbox';
                
                if (isNativeSelect && element.options) {{
                    // For select fields, show more options (up to 15) to help the agent choose from available options
                    // Filter out placeholder options like "Please select one", "Select one", etc.
                    const opts = Array.from(element.options)
                        .filter(opt => !opt.disabled)
                        .map(opt => (opt.textContent || opt.value || '').trim())
                        .filter(t => t && t !== 'Select one' && t !== 'Please select' && t !== '--' && t !== '')
                        .slice(0, 15);  // Show up to 15 options (increased from 6) to give agent more choices
                    if (opts.length > 0) {{
                        selectOptions = ' [options: ' + opts.join(', ') + (element.options.length > opts.length + 1 ? '...' : '') + ']';
                    }}
                }} else if (isCustomSelect) {{
                    // For custom selects, try to find options in the DOM if available
                    // Some custom selects have options visible in the DOM even when closed
                    try {{
                        const optionElements = element.querySelectorAll('[role="option"], option, li[data-value], div[data-value]');
                        if (optionElements.length > 0) {{
                            const opts = Array.from(optionElements)
                                .slice(0, 15)  // Show up to 15 options (increased from 6)
                                .map(opt => (opt.textContent || opt.getAttribute('data-value') || '').trim())
                                .filter(t => t && t !== 'Select one' && t !== 'Please select' && t !== '--' && t !== '');
                            if (opts.length > 0) {{
                                selectOptions = ' [options: ' + opts.join(', ') + (optionElements.length > opts.length ? '...' : '') + ']';
                            }}
                        }}
                    }} catch (e) {{
                        // If we can't extract options, still mark it as a select field
                        selectOptions = ' [custom select/dropdown]';
                    }}
                }}
                
                // Return element info with pre-calculated normalized coordinates
                return {{
                    index: index,
                    normalizedCoords: coords,
                    description: element.tagName.toLowerCase() + 
                                (element.textContent ? ': ' + element.textContent.trim().substring(0, 50) : '') +
                                (element.placeholder ? ' [' + element.placeholder + ']' : '') +
                                (element.type ? ' (type=' + element.type + ')' : '') +
                                selectOptions,
                    tagName: element.tagName.toLowerCase(),
                    type: element.type || '',
                    textContent: element.textContent?.trim().substring(0, 100) || '',
                    className: element.className || '',
                    id: element.id || '',
                    placeholder: element.placeholder || '',
                    role: element.getAttribute('role') || '',
                    name: element.getAttribute('name') || '',
                    ariaLabel: element.getAttribute('aria-label') || '',
                    href: element.getAttribute('href') || '',
                    fieldLabel: fieldLabel,  // Associated label/question text for form elements
                    contextText: collectContextText(element),
                    dataAttributes: collectDataAttributes(element),
                    selectOptions: selectOptions,  // Store select options
                    cssSelector: element.id ? '#' + element.id : 
                                element.getAttribute('name') ? '[name="' + element.getAttribute('name') + '"]' :
                                element.getAttribute('data-testid') ? '[data-testid="' + element.getAttribute('data-testid') + '"]' : null
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
                const style = window.getComputedStyle(element);
                if (style.display === 'none' || style.visibility === 'hidden' || 
                    element.offsetWidth === 0 || element.offsetHeight === 0) {{
                    return;
                }}
                
                // In interactive mode, filter aggressively to reduce noise
                if (!INCLUDE_ALL && !isLikelyInteractive(element)) {{
                    return;
                }}
                
                const rect = element.getBoundingClientRect();
                
                // Skip if element is completely outside the viewport (no intersection)
                if (rect.bottom <= 0 || rect.right <= 0 || rect.top >= viewportHeight || rect.left >= viewportWidth) {{
                    return;
                }}

                // Calculate normalized coordinates and validate them
                const normalizedCoords = [
                    Math.round((rect.top / viewportHeight) * 1000),     // y_min
                    Math.round((rect.left / viewportWidth) * 1000),     // x_min  
                    Math.round((rect.bottom / viewportHeight) * 1000),  // y_max
                    Math.round((rect.right / viewportWidth) * 1000)     // x_max
                ];

                // Skip elements with invalid coordinates (negative or out of bounds)
                if (normalizedCoords[0] < 0 || normalizedCoords[1] < 0 || 
                    normalizedCoords[2] > 1000 || normalizedCoords[3] > 1000 ||
                    normalizedCoords[0] > 1000 || normalizedCoords[1] > 1000) {{
                    // Skip this element silently - coordinates are invalid
                    return;
                }}
                
                // Only in interactive mode, drop tiny elements
                if (!INCLUDE_ALL) {{
                    if (rect.width < 10 || rect.height < 10) {{
                        return;
                    }}
                }}
                
                elementData.push(createNumberedOverlay(element, index, normalizedCoords));
                index++;
            }});
            
            console.log(`Created ${{elementData.length}} numbered overlays`);
            
            // Post-process: Find element relationships and group information
            // This helps the agent understand which elements belong together
            elementData.forEach((elem, idx) => {{
                const relationships = [];
                const elemName = elem.name || '';
                const elemFieldLabel = elem.fieldLabel || '';
                const elemType = elem.type || '';
                const elemTag = elem.tagName || '';
                
                // Find elements in the same group
                elementData.forEach((otherElem, otherIdx) => {{
                    if (idx === otherIdx) return;
                    
                    const otherName = otherElem.name || '';
                    const otherFieldLabel = otherElem.fieldLabel || '';
                    const otherType = otherElem.type || '';
                    const otherTag = otherElem.tagName || '';
                    
                    // Same radio group (same name attribute)
                    if (elemType === 'radio' && otherType === 'radio' && elemName && elemName === otherName) {{
                        relationships.push(otherIdx);
                    }}
                    // Same checkbox group (same name, though checkboxes can have multiple)
                    else if (elemType === 'checkbox' && otherType === 'checkbox' && elemName && elemName === otherName) {{
                        relationships.push(otherIdx);
                    }}
                    // Same question/field label (belong to same question)
                    else if (elemFieldLabel && elemFieldLabel === otherFieldLabel && elemFieldLabel.length > 10) {{
                        // Only group if they're form elements (inputs, radios, checkboxes, selects)
                        const isFormElement = ['input', 'select', 'textarea'].includes(elemTag) || 
                                            elemType === 'radio' || elemType === 'checkbox';
                        const isOtherFormElement = ['input', 'select', 'textarea'].includes(otherTag) || 
                                                   otherType === 'radio' || otherType === 'checkbox';
                        if (isFormElement && isOtherFormElement) {{
                            relationships.push(otherIdx);
                        }}
                    }}
                }});
                
                if (relationships.length > 0) {{
                    elem.relatedElements = relationships;
                    elem.groupSize = relationships.length + 1; // Include self
                }}
            }});
            
            return elementData;
        }})();
        """
        
        try:
            element_data = self.page.evaluate(js_code)
            try:
                from utils.event_logger import get_event_logger
                get_event_logger().system_debug(f"Created {len(element_data)} numbered element overlays")
            except Exception:
                pass
            return element_data or []
        except Exception as e:
            try:
                from utils.event_logger import get_event_logger
                get_event_logger().system_error("Error creating numbered overlays", error=e)
            except Exception:
                pass
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
                try:
                    from utils.event_logger import get_event_logger
                    get_event_logger().system_debug(f"Removed {count} element overlays")
                except Exception:
                    pass
        except Exception as e:
            try:
                from utils.event_logger import get_event_logger
                get_event_logger().system_error("Error removing element overlays", error=e)
            except Exception:
                pass
