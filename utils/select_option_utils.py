"""
Utility functions for selecting options from select fields.

This module provides comprehensive functionality for selecting options from both
native HTML select elements and custom dropdown implementations, with support
for lazy loading of options and closest match finding.
"""
import time
import difflib
from typing import List, Optional, Dict, Any
from playwright.sync_api import Page, ElementHandle


class SelectOptionError(Exception):
    """Raised when option selection fails."""
    pass


def select_option(
    page: Page,
    html_content: str,
    field_placeholder: str,
    option_text: str,
    timeout: float = 10.0,
    wait_for_options: bool = True,
    max_retries: int = 10,
    failed_selectors: Optional[List[str]] = None,
    clicked_element_info: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Universal dropdown selection using programmatic element discovery and option matching.

    This function works with any dropdown-like element (native select, custom div dropdowns, etc.)
    by clicking the trigger element and then finding/selecting the best matching option.

    Args:
        page: Playwright page instance
        html_content: Full HTML content of the webpage
        field_placeholder: Description/placeholder/label for the select field to target
        option_text: Text of the option to select (exact or closest match)
        timeout: Maximum time to wait for operations in seconds
        wait_for_options: Whether to wait for options to load after clicking
        max_retries: Maximum number of retry attempts
        failed_selectors: List of selectors that have already failed (internal use)
        clicked_element_info: Context about what element was clicked

    Returns:
        True if selection succeeded, False otherwise

    Raises:
        SelectOptionError: If selection fails after all attempts
    """
    failed_selectors = failed_selectors or []

    for attempt in range(max_retries):
        try:
            # Find a trigger element to click (could be any element that opens a dropdown)
            trigger_selector = _find_dropdown_trigger(
                page, html_content, field_placeholder, option_text, clicked_element_info, failed_selectors
            )

            if not trigger_selector:
                if attempt == max_retries - 1:
                    raise SelectOptionError(f"Could not find any dropdown trigger element for field '{field_placeholder}'")
                continue

            # Use universal dropdown selection algorithm
            return _universal_dropdown_select(
                page, trigger_selector, option_text, timeout
            )

        except Exception as e:
            error_msg = str(e)
            print(f"Attempt {attempt + 1} failed: {error_msg}")

            # For universal selection, we don't track failed selectors the same way
            # since we're not trying specific selectors but rather finding elements dynamically

            if attempt == max_retries - 1:
                raise SelectOptionError(f"Failed to select '{option_text}' after {max_retries} attempts")

            time.sleep(0.5)

    return False


def _select_option_attempt(
    page: Page,
    select_identifier: str,
    option_text: str,
    timeout: float,
    wait_for_options: bool
) -> bool:
    """Single attempt to select an option."""
    # Find the select element
    select_element = _find_select_element(page, select_identifier)
    if not select_element:
        raise SelectOptionError(f"Could not find select element: {select_identifier}")

    # Determine if it's a native select or custom dropdown
    is_native_select = _is_native_select(select_element)

    if is_native_select:
        return _select_native_option(select_element, option_text, timeout)
    else:
        return _select_custom_option(page, select_element, option_text, timeout, wait_for_options)


def _find_select_element(page: Page, identifier: str) -> Optional[ElementHandle]:
    """Find a select element by identifier."""
    # Try different selector strategies
    selectors = [
        identifier,  # Direct selector
        f"#{identifier}",  # ID selector
        f"[name='{identifier}']",  # Name attribute
        f"[data-testid='{identifier}']",  # Test ID
        f"[aria-label*='{identifier}']",  # ARIA label contains
    ]

    for selector in selectors:
        try:
            element = page.query_selector(selector)
            if element:
                return element
        except Exception:
            continue

    # Try to find by placeholder text (for cases where we only know the placeholder)
    try:
        element = page.query_selector(f"[placeholder*='{identifier}']")
        if element:
            return element
    except Exception:
        pass

    try:
        element = page.query_selector(f"input[placeholder*='{identifier}']")
        if element:
            return element
    except Exception:
        pass

    # Try to find select-like elements with placeholder-like text
    try:
        # Look for elements that contain the identifier text (might be placeholder or label)
        candidates = page.query_selector_all("select, input[role='combobox'], [role='combobox'], [role='listbox'], button[aria-haspopup], div[aria-haspopup]")
        for candidate in candidates:
            try:
                # Check various text sources on the element itself
                text_sources = [
                    candidate.get_attribute("placeholder"),
                    candidate.get_attribute("aria-label"),
                    candidate.text_content(),
                    candidate.get_attribute("title")
                ]

                for text in text_sources:
                    if text and identifier.lower() in text.lower():
                        return candidate
            except Exception:
                continue
    except Exception:
        pass

    # Also check associated labels
    try:
        # Find labels that contain the identifier text
        labels = page.query_selector_all("label")
        for label in labels:
            try:
                label_text = label.text_content()
                if label_text and identifier.lower() in label_text.lower():
                    # Get the 'for' attribute to find the associated element
                    for_attr = label.get_attribute("for")
                    if for_attr:
                        associated_element = page.query_selector(f"#{for_attr}")
                        if associated_element:
                            return associated_element

                    # If no 'for' attribute, check if the label contains a select/input element
                    child_element = label.query_selector("select, input, button, div[role]")
                    if child_element:
                        return child_element
            except Exception:
                continue
    except Exception:
        pass

    return None


def _is_native_select(element: ElementHandle) -> bool:
    """Check if element is a native HTML select."""
    try:
        tag_name = element.evaluate("el => el.tagName.toLowerCase()")
        return tag_name == "select"
    except Exception:
        return False


def _select_native_option(
    select_element: ElementHandle,
    option_text: str,
    timeout: float
) -> bool:
    """Select an option from a native HTML select element."""
    # Get available options
    options = select_element.evaluate("""
        el => Array.from(el.options || []).map(opt => ({
            value: opt.value,
            text: opt.textContent?.trim(),
            label: opt.label?.trim(),
            disabled: opt.disabled,
            selected: opt.selected
        }))
    """)

    if not options:
        raise SelectOptionError("No options found in select element")

    # Find best match
    best_match = _find_best_option_match(options, option_text)
    if not best_match:
        available_texts = [opt['text'] for opt in options if opt['text']]
        raise SelectOptionError(f"No matching option found. Available: {available_texts}")

    # Select the option
    select_element.select_option(value=best_match['value'])
    return True


def _select_custom_option(
    page: Page,
    select_element: ElementHandle,
    option_text: str,
    timeout: float,
    wait_for_options: bool
) -> bool:
    """Select an option from a custom dropdown implementation."""
    # Check if this is an input element (searchable combobox)
    try:
        tag_name = select_element.evaluate("el => el.tagName.toLowerCase()")
        role = select_element.evaluate("el => el.getAttribute('role') || ''").lower()

        is_input_element = tag_name == "input"
        is_searchable = is_input_element or role == "combobox"

        if is_searchable and option_text:
            return _handle_searchable_combobox(page, select_element, option_text, timeout)
    except Exception:
        # If we can't determine the element type, fall back to regular handling
        pass

    # Regular custom dropdown handling
    # Click to open the dropdown
    select_element.click()

    # Wait for options to appear if requested
    if wait_for_options:
        _wait_for_options_to_load(page, timeout)

    # Find option elements
    option_elements = _find_option_elements(page, select_element, option_text)

    if not option_elements:
        raise SelectOptionError(f"No option elements found for '{option_text}'")

    # Find the best matching option element
    best_option = _find_best_option_element(option_elements, option_text)

    if not best_option:
        raise SelectOptionError(f"No matching option element found for '{option_text}'")

    # Click the option
    best_option.click()
    return True


def _handle_searchable_combobox(
    page: Page,
    input_element: ElementHandle,
    option_text: str,
    timeout: float
) -> bool:
    """Handle searchable combobox (input element with dropdown)."""
    try:
        # First, try to find all available options (open the dropdown fully)
        input_element.click()
        time.sleep(0.2)

        # Get all available options from the dropdown
        all_options = []
        option_selectors = [
            '[role="option"]',
            'li[data-value]',
            'div[data-value]',
            'li',
            'div[onclick]',
        ]

        for selector in option_selectors:
            try:
                options = page.query_selector_all(selector)
                for option in options:
                    try:
                        text_content = option.evaluate("el => el.textContent?.trim() || ''")
                        if text_content:
                            all_options.append((option, text_content))
                    except Exception:
                        continue
            except Exception:
                continue

        if not all_options:
            # Fallback: just fill the input and press enter
            input_element.fill("")
            input_element.fill(option_text)
            input_element.press("Enter")
            return True

        # Find the best matching option
        best_option = None
        best_score = 0

        for option, text_content in all_options:
            score = _calculate_text_similarity(text_content, option_text)
            if score > best_score:
                best_score = score
                best_option = option

        if best_option:
            # Fill the input with the exact option text
            input_element.fill("")
            exact_text = None
            for opt, text in all_options:
                if opt == best_option:
                    exact_text = text
                    break

            if exact_text:
                input_element.fill(exact_text)
                # Try to click the option if it's still available
                try:
                    best_option.click()
                except Exception:
                    pass  # Option might not be clickable after filling
                return True

        # Fallback: just fill with the requested text and press enter
        input_element.fill("")
        input_element.fill(option_text)
        input_element.press("Enter")
        return True

    except Exception as e:
        raise SelectOptionError(f"Failed to handle searchable combobox: {e}")


def _wait_for_options_to_load(page: Page, timeout: float):
    """Wait for dropdown options to become visible."""
    start_time = time.time()

    while time.time() - start_time < timeout:
        # Check for common option selectors
        option_selectors = [
            '[role="option"]',
            '[role="listbox"] [role="option"]',
            '.dropdown-option',
            '.select-option',
            'li[role="option"]',
            'div[role="option"]',
            'button[data-value]',
            'li[data-value]',
            'div[data-value]',
        ]

        for selector in option_selectors:
            try:
                options = page.query_selector_all(selector)
                if options and len(options) > 0:
                    # Wait a bit more to ensure they're fully loaded
                    time.sleep(0.2)
                    return
            except Exception:
                continue

        # Check for visible dropdown/listbox containers
        try:
            visible_containers = page.evaluate("""
                () => {
                    const containers = document.querySelectorAll('[role="listbox"], [role="menu"], .dropdown, .select-dropdown');
                    return Array.from(containers).filter(el =>
                        el.offsetWidth > 0 &&
                        el.offsetHeight > 0 &&
                        getComputedStyle(el).display !== 'none' &&
                        getComputedStyle(el).visibility !== 'hidden'
                    ).length;
                }
            """)
            if visible_containers > 0:
                time.sleep(0.2)
                return
        except Exception:
            pass

        time.sleep(0.1)

    # If we get here, options may not have loaded, but we'll continue anyway


def _find_option_elements(
    page: Page,
    select_element: ElementHandle,
    option_text: str
) -> List[ElementHandle]:
    """Find option elements in a custom dropdown."""
    option_elements = []

    # Common selectors for option elements
    option_selectors = [
        '[role="option"]',
        '[role="listbox"] [role="option"]',
        '.dropdown-option',
        '.select-option',
        'li[data-value]',
        'div[data-value]',
        'button[data-value]',
        'li',
        'div[onclick]',
        'button',
    ]

    for selector in option_selectors:
        try:
            elements = page.query_selector_all(selector)
            if elements:
                option_elements.extend(elements)
        except Exception:
            continue

    # Filter to elements that are currently visible and contain relevant text
    visible_options = []
    for element in option_elements:
        try:
            is_visible = element.evaluate("""
                el => {
                    const rect = el.getBoundingClientRect();
                    const style = getComputedStyle(el);
                    return rect.width > 0 &&
                           rect.height > 0 &&
                           style.display !== 'none' &&
                           style.visibility !== 'hidden' &&
                           el.offsetParent !== null;
                }
            """)

            if is_visible:
                text_content = element.evaluate("el => el.textContent?.trim() || ''")
                if text_content:
                    visible_options.append(element)
        except Exception:
            continue

    return visible_options


def _find_best_option_element(
    option_elements: List[ElementHandle],
    target_text: str
) -> Optional[ElementHandle]:
    """Find the best matching option element."""
    best_match = None
    best_score = 0

    for element in option_elements:
        try:
            text_content = element.evaluate("el => el.textContent?.trim() || ''")
            if not text_content:
                continue

            # Calculate similarity score
            score = _calculate_text_similarity(text_content, target_text)

            if score > best_score:
                best_score = score
                best_match = element

        except Exception:
            continue

    return best_match


def _find_best_option_match(options: List[Dict[str, Any]], target_text: str) -> Optional[Dict[str, Any]]:
    """Find the best matching option from a list of option data."""
    best_match = None
    best_score = 0

    for option in options:
        if option.get('disabled'):
            continue

        # Check text content, label, and value
        texts_to_check = [
            option.get('text', ''),
            option.get('label', ''),
            option.get('value', ''),
        ]

        for text in texts_to_check:
            if text:
                score = _calculate_text_similarity(text, target_text)
                if score > best_score:
                    best_score = score
                    best_match = option

    return best_match


def _calculate_text_similarity(text1: str, text2: str) -> float:
    """Calculate similarity score between two texts."""
    if not text1 or not text2:
        return 0.0

    text1 = text1.lower().strip()
    text2 = text2.lower().strip()

    # Exact match gets highest score
    if text1 == text2:
        return 1.0

    # Contains match gets high score
    if text2 in text1 or text1 in text2:
        return 0.8

    # Use difflib for fuzzy matching
    similarity = difflib.SequenceMatcher(None, text1, text2).ratio()

    # Boost score for partial matches
    if similarity > 0.6:
        return similarity

    # Check word overlap
    words1 = set(text1.split())
    words2 = set(text2.split())
    if words1 & words2:
        overlap_ratio = len(words1 & words2) / max(len(words1), len(words2))
        return overlap_ratio * 0.5

    return similarity


def select_option_by_placeholder(
    page: Page,
    placeholder_text: str,
    option_text: str,
    timeout: float = 10.0,
    wait_for_options: bool = True,
    max_retries: int = 10
) -> bool:
    """
    Select an option from a select field by its placeholder text.

    Args:
        page: Playwright page instance
        placeholder_text: The placeholder text of the select/input field
        option_text: Text of the option to select (exact or closest match)
        timeout: Maximum time to wait for operations in seconds
        wait_for_options: Whether to wait for options to load after clicking
        max_retries: Maximum number of retry attempts

    Returns:
        True if selection succeeded, False otherwise

    Raises:
        SelectOptionError: If selection fails after all attempts
    """
    return select_option(
        page=page,
        select_identifier=placeholder_text,
        option_text=option_text,
        timeout=timeout,
        wait_for_options=wait_for_options,
        max_retries=max_retries
    )


def find_element_by_placeholder(page: Page, placeholder_text: str) -> Optional[ElementHandle]:
    """
    Find a form element by its placeholder text.

    Args:
        page: Playwright page instance
        placeholder_text: The placeholder text to search for

    Returns:
        ElementHandle if found, None otherwise
    """
    return _find_select_element(page, placeholder_text)


def _find_dropdown_trigger(
    page: Page,
    html_content: str,
    field_placeholder: str,
    option_text: str,
    clicked_element_info: Optional[Dict[str, Any]] = None,
    failed_selectors: Optional[List[str]] = None
) -> Optional[str]:
    """
    Find any element that could trigger a dropdown (select, input, div, button, etc.)
    using programmatic discovery based on context.
    """
    failed_selectors = failed_selectors or []

    # Strategy 1: Use programmatic element discovery with context
    print("🔍 Finding dropdown trigger element...")
    trigger_selector = _find_trigger_programmatically(page, field_placeholder, option_text, clicked_element_info, failed_selectors)

    if trigger_selector:
        return trigger_selector

    # Strategy 2: Fallback to LLM-based element finding
    print("🤖 Using LLM to find dropdown trigger...")
    trigger_selector = _find_trigger_with_llm(html_content, field_placeholder, option_text, failed_selectors)

    return trigger_selector


def _find_trigger_programmatically(
    page: Page,
    field_placeholder: str,
    option_text: str,
    clicked_element_info: Optional[Dict[str, Any]] = None,
    failed_selectors: Optional[List[str]] = None
) -> Optional[str]:
    """
    Programmatically find elements that could trigger dropdowns based on context.
    """
    failed_selectors = failed_selectors or []

    # Strategy 1: Use clicked element context for targeted search
    if clicked_element_info:
        keywords = clicked_element_info.get("keywords", [])
        for keyword in keywords[:5]:
            if len(keyword) > 2:
                # Look for various element types that could be dropdown triggers
                trigger_patterns = [
                    f"[aria-label*='{keyword}']",
                    f"[placeholder*='{keyword}']",
                    f"[title*='{keyword}']",
                    f"[name*='{keyword}']",
                    f"[id*='{keyword}']",
                    f"[class*='{keyword}']",
                ]

                for pattern in trigger_patterns:
                    if pattern not in failed_selectors:
                        try:
                            elements = page.locator(pattern).all()
                            for elem in elements[:3]:  # Limit per pattern
                                # Check if this element looks like a dropdown trigger
                                tag_name = elem.evaluate("el => el.tagName.toLowerCase()")
                                role = elem.get_attribute("role")
                                aria_expanded = elem.get_attribute("aria-expanded")
                                aria_haspopup = elem.get_attribute("aria-haspopup")

                                if (tag_name in ["select", "input", "button", "div"] or
                                    role in ["combobox", "button", "textbox"] or
                                    aria_expanded is not None or
                                    aria_haspopup in ["listbox", "menu", "true"]):
                                    elem_id = elem.get_attribute("id")
                                    if elem_id and f"#{elem_id}" not in failed_selectors:
                                        try:
                                            test_elements = page.locator(f"#{elem_id}").all()
                                            if len(test_elements) >= 1:
                                                print(f"✅ Found trigger element with keyword '{keyword}': ID '{elem_id}' matches {len(test_elements)} element(s)")
                                                return f"#{elem_id}"
                                        except Exception:
                                            continue
                        except Exception:
                            continue

    # Strategy 2: Look for common dropdown trigger patterns
    trigger_selectors = [
        # Native select elements
        "select[id]",
        "[role='combobox'][id]",
        "[role='listbox'][id]",
        # Input elements that might have dropdowns
        "input[list][id]",  # datalist inputs
        "input[aria-haspopup][id]",
        "input[aria-expanded][id]",
        # Button/div elements that trigger dropdowns
        "[aria-haspopup='listbox'][id]",
        "[aria-expanded][id]",
        "[data-dropdown][id]",
        "button[id]",
        "div[role='button'][id]",
        # SAP UI5 and other framework specific
        ".sapMSelect[id]",
        "[data-sap-ui][id]",
    ]

    for selector_pattern in trigger_selectors:
        try:
            elements = page.locator(selector_pattern).all()
            for element in elements[:5]:  # Limit to avoid too many checks
                try:
                    element_id = element.get_attribute("id")
                    if element_id and f"#{element_id}" not in failed_selectors:
                        # Validate the element exists and is clickable
                        try:
                            test_elements = page.locator(f"#{element_id}").all()
                            if len(test_elements) >= 1:
                                print(f"✅ Found trigger element with pattern '{selector_pattern}': ID '{element_id}' matches {len(test_elements)} element(s)")
                                return f"#{element_id}"
                        except Exception:
                            continue
                except Exception:
                    continue
        except Exception:
            continue

    print("❌ No suitable trigger element found programmatically")
    return None


def _find_trigger_with_llm(
    html_content: str,
    field_placeholder: str,
    option_text: str,
    failed_selectors: Optional[List[str]] = None
) -> Optional[str]:
    """
    Use LLM to find a dropdown trigger element from HTML content.
    """
    failed_selectors = failed_selectors or []

    try:
        from ai_utils import generate_text

        # Truncate HTML content if too long
        truncated_html = html_content[:10000] + "..." if len(html_content) > 10000 else html_content

        failed_list = ", ".join(f"'{s}'" for s in failed_selectors)

        prompt = f"""
Find a CSS selector for a dropdown trigger element (could be select, input, button, div, etc.) that would open a dropdown containing options including "{option_text}".

Field context: {field_placeholder}
Target option: {option_text}
Failed selectors: {failed_list}

Analyze the HTML below and find any element that could trigger a dropdown with options. Look for:
- select elements
- input elements with dropdown behavior
- buttons or divs with aria attributes indicating dropdown functionality
- elements with roles like combobox, button, etc.

HTML Content:
{truncated_html}

Return only a valid CSS selector (like '#element-id', '.class-name', '[attribute="value"]', or 'tagname'). Do not include any explanation.
"""

        trigger_selector = generate_text(
            prompt=prompt,
            model="gpt-5-mini",
            reasoning_level="low"
        ).strip()

        # Clean up the response
        trigger_selector = trigger_selector.strip('`"\'')
        return trigger_selector if trigger_selector else None

    except Exception as e:
        print(f"⚠️ Failed to find trigger with LLM: {e}")
        return None


def _get_selector_from_html_and_placeholder(
    html_content: str,
    field_placeholder: str,
    option_text: str
) -> Optional[str]:
    """Get a CSS selector for a select field using HTML content and field description."""
    try:
        from ai_utils import generate_text

        # Truncate HTML content if too long (keep first 10k chars, should be enough for most pages)
        truncated_html = html_content[:10000] + "..." if len(html_content) > 10000 else html_content

        prompt = f"""
I need to find a CSS selector for a dropdown/select element based on the HTML content and field description.

Field Description: {field_placeholder}
Option to select: {option_text}

HTML Content:
{truncated_html}

Find the most appropriate CSS selector for the select/dropdown element that matches this field description.
Return only a valid CSS selector (like '#element-id', '.class-name', '[attribute="value"]', or 'tagname').
Do not include any explanation or additional text.
"""

        selector = generate_text(
            prompt=prompt,
            model="gpt-5-mini",
            reasoning_level="low"
        ).strip()

        # Clean up the response (remove quotes, extra whitespace)
        selector = selector.strip('`"\'')
        return selector if selector else None

    except Exception as e:
        print(f"Error getting selector from HTML: {e}")
        return None


def _get_alternative_selector_from_llm(
    page: Page,
    option_text: str,
    failed_selectors: List[str],
    html_content: Optional[str] = None,
    clicked_element_info: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """Find an alternative selector by programmatically searching for dropdown elements."""
    try:
        # Try programmatic approaches first
        print("🔍 Trying programmatic element discovery...")

        # Strategy 1: Find all select elements and try them
        try:
            select_elements = page.locator("select").all()
            for select_elem in select_elements[:10]:  # Limit to first 10 to avoid too many attempts
                try:
                    element_id = select_elem.get_attribute("id")
                    element_name = select_elem.get_attribute("name")
                    element_class = select_elem.get_attribute("class")
                    aria_label = select_elem.get_attribute("aria-label")

                    # Try different selector strategies
                    candidates = []
                    if element_id:
                        candidates.append(f"#{element_id}")
                    if element_name:
                        candidates.append(f"[name='{element_name}']")
                    if aria_label:
                        candidates.append(f"[aria-label='{aria_label}']")
                    if element_class:
                        # Try the first class
                        first_class = element_class.split()[0]
                        candidates.append(f"select.{first_class}")

                    # Test each candidate
                    for selector in candidates:
                        if selector not in failed_selectors:
                            try:
                                elements = page.locator(selector).all()
                                if len(elements) >= 1:
                                    print(f"✅ Found select element with selector '{selector}' matches {len(elements)} element(s)")
                                    return selector
                            except Exception:
                                continue
                except Exception:
                    continue
        except Exception as e:
            print(f"⚠️ Error searching select elements: {e}")

        # Strategy 2: Find combobox/listbox elements
        try:
            combobox_selectors = [
                "[role='combobox']",
                "[role='listbox']",
                "[aria-haspopup='listbox']",
                "[aria-expanded]",
                ".sapMSelect",  # SAP UI5 specific
                "[data-sap-ui]",  # SAP UI5 specific
            ]

            for combo_selector in combobox_selectors:
                try:
                    elements = page.locator(combo_selector).all()
                    for element in elements[:5]:  # Limit per selector type
                        try:
                            element_id = element.get_attribute("id")
                            element_class = element.get_attribute("class")
                            aria_label = element.get_attribute("aria-label")

                            candidates = []
                            if element_id:
                                candidates.append(f"#{element_id}")
                            if aria_label:
                                candidates.append(f"[aria-label='{aria_label}']")
                            if element_class:
                                first_class = element_class.split()[0]
                                candidates.append(f"{combo_selector}.{first_class}")

                            for selector in candidates:
                                if selector not in failed_selectors:
                                    try:
                                        test_elements = page.locator(selector).all()
                                        if len(test_elements) >= 1:
                                            print(f"✅ Found combobox element with selector '{selector}' matches {len(test_elements)} element(s)")
                                            return selector
                                    except Exception:
                                        continue
                        except Exception:
                            continue
                except Exception:
                    continue
        except Exception as e:
            print(f"⚠️ Error searching combobox elements: {e}")

        # Strategy 3: Look for elements containing country/region related text
        try:
            # Look for elements that might contain country/region related content
            country_indicators = ["country", "region", "phone", "dial", "code"]
            for indicator in country_indicators:
                try:
                    # Find elements containing these keywords in attributes or nearby text
                    selector_patterns = [
                        f"[aria-label*='{indicator}']",
                        f"[placeholder*='{indicator}']",
                        f"[name*='{indicator}']",
                        f"[id*='{indicator}']",
                        f"[class*='{indicator}']",
                    ]

                    for pattern in selector_patterns:
                        if pattern not in failed_selectors:
                            try:
                                elements = page.locator(pattern).all()
                                for element in elements[:3]:  # Limit per pattern
                                    try:
                                        # Check if this element or its parent looks like a dropdown
                                        tag_name = element.evaluate("el => el.tagName.toLowerCase()")
                                        role = element.get_attribute("role")
                                        aria_expanded = element.get_attribute("aria-expanded")

                                        if (tag_name in ["select", "input", "button", "div"] or
                                            role in ["combobox", "listbox"] or
                                            aria_expanded is not None):
                                            print(f"✅ Found element with {indicator} indicator: '{pattern}' matches {len(elements)} element(s)")
                                            return pattern
                                    except Exception:
                                        continue
                            except Exception:
                                continue
                except Exception:
                    continue
        except Exception as e:
            print(f"⚠️ Error searching for country indicators: {e}")

        # Strategy 4: If we have clicked element info, look for related elements
        if clicked_element_info:
            try:
                print("🎯 Using clicked element context for targeted search...")
                clicked_text = clicked_element_info.get("text", "").lower()
                keywords = clicked_element_info.get("keywords", [])

                # Look for elements containing keywords from the clicked element
                for keyword in keywords[:5]:  # Limit to first 5 keywords
                    if len(keyword) > 2:  # Meaningful keywords only
                        try:
                            # Search for various attributes that might contain the keyword
                            search_patterns = [
                                f"[aria-label*='{keyword}']",
                                f"[placeholder*='{keyword}']",
                                f"[title*='{keyword}']",
                                f"[name*='{keyword}']",
                                f"[id*='{keyword}']",
                                f"[class*='{keyword}']",
                            ]

                            for pattern in search_patterns:
                                if pattern not in failed_selectors:
                                    try:
                                        elements = page.locator(pattern).all()
                                        for elem in elements[:3]:  # Limit per pattern
                                            # Check if this element looks like a dropdown/select
                                            tag_name = elem.evaluate("el => el.tagName.toLowerCase()")
                                            role = elem.get_attribute("role")
                                            aria_expanded = elem.get_attribute("aria-expanded")

                                            if (tag_name == "select" or
                                                role in ["combobox", "listbox"] or
                                                aria_expanded is not None or
                                                tag_name in ["input", "button", "div"]):
                                                elem_id = elem.get_attribute("id")
                                                if elem_id and f"#{elem_id}" not in failed_selectors:
                                                    try:
                                                        test_elements = page.locator(f"#{elem_id}").all()
                                                        if len(test_elements) >= 1:
                                                            print(f"✅ Found element with keyword '{keyword}': ID '{elem_id}' matches {len(test_elements)} element(s)")
                                                            return f"#{elem_id}"
                                                    except Exception:
                                                        continue
                                                # If no ID, try other selectors
                                                elem_name = elem.get_attribute("name")
                                                if elem_name and f"[name='{elem_name}']" not in failed_selectors:
                                                    try:
                                                        test_elements = page.locator(f"[name='{elem_name}']").all()
                                                        if len(test_elements) >= 1:
                                                            print(f"✅ Found element with keyword '{keyword}': name '{elem_name}' matches {len(test_elements)} element(s)")
                                                            return f"[name='{elem_name}']"
                                                    except Exception:
                                                        continue
                                    except Exception:
                                        continue
                        except Exception:
                            continue

                # If no specific matches found with keywords, try broader text matching
                if clicked_text:
                    # Look for elements near text that matches our clicked description
                    broad_keywords = ["country", "region", "phone", "select", "choose", "dropdown"]
                    for broad_keyword in broad_keywords:
                        if broad_keyword in clicked_text:
                            try:
                                broad_patterns = [
                                    f"[aria-label*='{broad_keyword}']",
                                    f"[placeholder*='{broad_keyword}']",
                                    f"[id*='{broad_keyword}']",
                                ]
                                for pattern in broad_patterns:
                                    if pattern not in failed_selectors:
                                        try:
                                            elements = page.locator(pattern).all()
                                            for elem in elements[:2]:
                                                tag_name = elem.evaluate("el => el.tagName.toLowerCase()")
                                                role = elem.get_attribute("role")
                                                if (tag_name == "select" or
                                                    role in ["combobox", "listbox"] or
                                                    elem.get_attribute("aria-expanded") is not None):
                                                    elem_id = elem.get_attribute("id")
                                                    if elem_id and f"#{elem_id}" not in failed_selectors:
                                                        try:
                                                            test_elements = page.locator(f"#{elem_id}").all()
                                                            if len(test_elements) >= 1:
                                                                print(f"✅ Found broad match for '{broad_keyword}': ID '{elem_id}' matches {len(test_elements)} element(s)")
                                                                return f"#{elem_id}"
                                                        except Exception:
                                                            continue
                                        except Exception:
                                            continue
                            except Exception:
                                continue

            except Exception as e:
                print(f"⚠️ Error in context-aware search: {e}")

        print("❌ All programmatic approaches failed to find a suitable element")
        return None

    except Exception as e:
        print(f"⚠️ Failed to get alternative selector: {e}")

    return None


def find_closest_option_text(
    page: Page,
    select_identifier: str,
    target_text: str
) -> Optional[str]:
    """
    Find the closest matching option text for a select field.

    Args:
        page: Playwright page instance
        select_identifier: CSS selector or identifier for the select field
        target_text: Target option text to find closest match for

    Returns:
        The closest matching option text, or None if no close match found
    """
    try:
        select_element = _find_select_element(page, select_identifier)
        if not select_element:
            return None

        if _is_native_select(select_element):
            # Get options from native select
            options = select_element.evaluate("""
                el => Array.from(el.options || []).map(opt => ({
                    value: opt.value,
                    text: opt.textContent?.trim(),
                    label: opt.label?.trim()
                }))
            """)
        else:
            # For custom selects, we need to click and find options
            select_element.click()
            time.sleep(0.5)  # Wait for options

            option_elements = _find_option_elements(page, select_element, "")
            options = []
            for element in option_elements:
                try:
                    text = element.evaluate("el => el.textContent?.trim() || ''")
                    if text:
                        options.append({'text': text})
                except Exception:
                    continue

        if not options:
            return None

        best_match = _find_best_option_match(options, target_text)
        return best_match['text'] if best_match else None

    except Exception:
        return None


def _universal_dropdown_select(
    page: Page,
    trigger_selector: str,
    option_text: str,
    timeout: float = 30.0
) -> bool:
    """
    Universal dropdown selection algorithm that works with any dropdown-like element.

    1. Click the trigger element to open the dropdown
    2. Find all available options
    3. Select the closest match to the desired option
    """
    try:
        print(f"🎯 Using universal dropdown selection for trigger: '{trigger_selector}'")

        # Click the trigger element to open the dropdown
        print("🖱️ Clicking trigger element...")
        trigger_element = page.locator(trigger_selector).first
        trigger_element.click()

        # Wait for dropdown options to appear
        print("📋 Waiting for dropdown options...")
        time.sleep(1.0)  # Give time for options to load

        # Find all potential option elements using various strategies
        option_elements = _find_dropdown_options(page, trigger_selector)

        if not option_elements:
            print("❌ No dropdown options found")
            return False

        print(f"📋 Found {len(option_elements)} potential options")

        # Extract text from each option
        options_data = []
        for i, elem in enumerate(option_elements):
            try:
                # Get option text using various methods
                text = elem.evaluate("""
                    el => {
                        // Try different ways to get option text
                        return el.textContent?.trim() ||
                               el.innerText?.trim() ||
                               el.getAttribute('value') ||
                               el.getAttribute('data-value') ||
                               '';
                    }
                """)
                if text:
                    options_data.append({
                        'element': elem,
                        'text': text,
                        'index': i
                    })
            except Exception:
                continue

        if not options_data:
            print("❌ No valid option texts found")
            return False

        # Find the best matching option
        best_match = _find_best_option_match(options_data, option_text)

        if not best_match:
            print(f"❌ No good match found for '{option_text}'")
            return False

        print(f"🎯 Best match: '{best_match['text']}' (score: {best_match.get('score', 'N/A')})")

        # Click the selected option
        print("🖱️ Clicking selected option...")
        best_match['element'].click()

        # Wait a moment for the selection to take effect
        time.sleep(0.5)

        print("✅ Successfully completed universal dropdown selection!")
        return True

    except Exception as e:
        print(f"❌ Universal dropdown selection failed: {e}")
        return False


def _find_dropdown_options(page: Page, trigger_selector: str) -> List[ElementHandle]:
    """
    Find all option elements that appear after clicking a dropdown trigger.
    Uses multiple strategies to locate options.
    """
    option_elements = []

    # Strategy 1: Look for common option selectors
    option_selectors = [
        "[role='option']",
        "[role='menuitem']",
        "option",
        ".option",
        "[data-option]",
        "[data-value]",
        "li",
        ".dropdown-item",
        ".select-option",
        ".menu-item",
    ]

    for selector in option_selectors:
        try:
            elements = page.locator(selector).all()
            if elements:
                option_elements.extend(elements)
        except Exception:
            continue

    # Strategy 2: Look for elements that appeared after clicking (new elements)
    # This is more complex and might not be reliable, so we'll stick with the above for now

    # Remove duplicates while preserving order
    seen = set()
    unique_elements = []
    for elem in option_elements:
        try:
            # Use element identity as unique key
            elem_id = elem.evaluate("el => el.id || el.className || el.textContent?.slice(0,50)")
            if elem_id not in seen:
                seen.add(elem_id)
                unique_elements.append(elem)
        except Exception:
            unique_elements.append(elem)

    return unique_elements[:50]  # Limit to reasonable number
