"""
Test smart field handling for select, upload, and date inputs.
This demonstrates a robust approach that works without relying on visibility or clicking options.
"""
import time
from pathlib import Path
from typing import Optional

import pytest
from playwright.sync_api import sync_playwright, Page


class SmartFieldHandler:
    """Handles select, upload, and date fields programmatically"""
    
    def __init__(self, page: Page):
        self.page = page
    
    def handle_select(
        self, 
        selector: str, 
        option_value: Optional[str] = None,
        option_text: Optional[str] = None
    ) -> dict:
        """
        Handle select field by programmatically getting options and setting value.
        Works for both native <select> and custom dropdowns.
        
        Args:
            selector: CSS selector for the select element or trigger
            option_value: Value to select (if known)
            option_text: Text to match (if value not known)
            
        Returns:
            Dict with success status and selected option info
        """
        # Step 1: Determine if native or custom
        element_info = self.page.evaluate("""
            (sel) => {
                const el = document.querySelector(sel);
                if (!el) return null;
                return {
                    tag: el.tagName.toLowerCase(),
                    role: el.getAttribute('role') || '',
                    type: el.type || ''
                };
            }
        """, selector)
        
        if not element_info:
            return {"success": False, "error": "Element not found"}
        
        is_native = element_info['tag'] == 'select'
        
        if is_native:
            return self._handle_native_select(selector, option_value, option_text)
        else:
            return self._handle_custom_select(selector, option_value, option_text)
    
    def _handle_native_select(
        self, 
        selector: str, 
        option_value: Optional[str],
        option_text: Optional[str]
    ) -> dict:
        """Handle native <select> element programmatically"""
        # Get all options (including ones outside viewport)
        options = self.page.evaluate("""
            (sel) => {
                const select = document.querySelector(sel);
                if (!select) return null;
                return {
                    options: Array.from(select.options || []).map(opt => ({
                        value: opt.value || '',
                        text: (opt.textContent || '').trim(),
                        label: opt.label || '',
                        index: opt.index,
                        disabled: opt.disabled,
                        selected: opt.selected
                    })),
                    currentValue: select.value,
                    currentText: select.options[select.selectedIndex]?.textContent?.trim() || ''
                };
            }
        """, selector)
        
        if not options or not options['options']:
            return {"success": False, "error": "No options found"}
        
        print(f"  📋 Found {len(options['options'])} options in select")
        print(f"  📋 Current: value='{options['currentValue']}', text='{options['currentText']}'")
        
        # Find target option
        target_option = None
        
        # Strategy 1: Match by value
        if option_value:
            for opt in options['options']:
                if not opt['disabled'] and opt['value'] == option_value:
                    target_option = opt
                    break
        
        # Strategy 2: Match by text (partial or full)
        if not target_option and option_text:
            for opt in options['options']:
                if not opt['disabled']:
                    if option_text.lower() in opt['text'].lower():
                        target_option = opt
                        break
        
        # Strategy 3: No option specified - pick first non-placeholder
        if not target_option:
            for opt in options['options']:
                if not opt['disabled']:
                    # Skip first if it has empty value (likely placeholder)
                    if opt['index'] == 0 and opt['value'].strip() == '':
                        continue
                    target_option = opt
                    break
        
        if not target_option:
            return {"success": False, "error": "No suitable option found"}
        
        # Set the value programmatically and dispatch events
        result = self.page.evaluate("""
            (args) => {
                const select = document.querySelector(args.selector);
                if (!select) return false;
                
                select.value = args.value;
                select.dispatchEvent(new Event('input', { bubbles: true }));
                select.dispatchEvent(new Event('change', { bubbles: true }));
                select.dispatchEvent(new Event('blur', { bubbles: true }));
                
                return {
                    value: select.value,
                    text: select.options[select.selectedIndex]?.textContent?.trim() || ''
                };
            }
        """, {"selector": selector, "value": target_option['value']})
        
        print(f"  ✅ Selected: '{target_option['text']}' (value='{target_option['value']}')")
        return {"success": True, "selected": result}
    
    def _handle_custom_select(
        self,
        selector: str,
        option_value: Optional[str],
        option_text: Optional[str]
    ) -> dict:
        """Handle custom dropdown/listbox/combobox programmatically"""
        # Click trigger to open
        try:
            self.page.click(selector)
            time.sleep(0.3)  # Wait for options to appear
        except Exception as e:
            print(f"  ⚠️ Click failed: {e}")
        
        # Get all visible options
        options = self.page.evaluate("""
            () => {
                const opts = document.querySelectorAll('[role="option"]');
                return Array.from(opts).map((opt, idx) => ({
                    text: (opt.textContent || '').trim(),
                    value: opt.getAttribute('data-value') || opt.textContent?.trim() || '',
                    index: idx,
                    visible: opt.offsetParent !== null,
                    selector: opt.id ? `#${opt.id}` : null
                })).filter(o => o.visible);
            }
        """)
        
        print(f"  📋 Found {len(options)} visible custom options")
        
        # Find target
        target = None
        if option_text:
            for opt in options:
                if option_text.lower() in opt['text'].lower():
                    target = opt
                    break
        elif option_value:
            for opt in options:
                if opt['value'] == option_value:
                    target = opt
                    break
        
        # Fallback: pick first non-placeholder
        if not target and options:
            # Skip first if text looks like placeholder
            target = options[1] if len(options) > 1 else options[0]
        
        if not target:
            return {"success": False, "error": "No option found"}
        
        # Click the option
        try:
            # Try to click by text
            self.page.click(f'[role="option"]:has-text("{target["text"]}")')
            print(f"  ✅ Clicked custom option: '{target['text']}'")
            return {"success": True, "selected": target}
        except Exception as e:
            print(f"  ❌ Failed to click option: {e}")
            return {"success": False, "error": str(e)}
    
    def handle_upload(self, selector: str, file_path: str) -> dict:
        """
        Handle file upload programmatically.
        Tries direct file set first, then click + set if needed.
        
        Args:
            selector: CSS selector for input[type=file] or upload trigger
            file_path: Path to file to upload
        """
        # Step 1: Try to find the actual file input
        input_selector = self.page.evaluate("""
            (sel) => {
                let el = document.querySelector(sel);
                if (!el) return null;
                
                // If it's already a file input, use it
                if (el.tagName.toLowerCase() === 'input' && el.type === 'file') {
                    return sel;
                }
                
                // Look for file input nearby or inside
                const fileInput = el.querySelector('input[type="file"]') || 
                                 el.parentElement?.querySelector('input[type="file"]');
                
                if (fileInput) {
                    return fileInput.id ? `#${fileInput.id}` : 
                           fileInput.name ? `[name="${fileInput.name}"]` : null;
                }
                
                return null;
            }
        """, selector)
        
        # Strategy 1: Set file directly on input
        if input_selector:
            try:
                self.page.set_input_files(input_selector, file_path)
                print(f"  ✅ Uploaded file directly: {Path(file_path).name}")
                return {"success": True, "method": "direct"}
            except Exception as e:
                print(f"  ⚠️ Direct upload failed: {e}")
        
        # Strategy 2: Click trigger first, then find and set input
        try:
            self.page.click(selector)
            time.sleep(0.2)
            
            # Try to find file input after click
            file_input = self.page.locator('input[type="file"]').first
            if file_input.count() > 0:
                file_input.set_input_files(file_path)
                print(f"  ✅ Uploaded file after click: {Path(file_path).name}")
                return {"success": True, "method": "click_then_set"}
        except Exception as e:
            print(f"  ❌ Upload failed: {e}")
            return {"success": False, "error": str(e)}
        
        return {"success": False, "error": "Could not find file input"}
    
    def handle_datetime(
        self,
        selector: str,
        value: str,
        input_type: str = "date"
    ) -> dict:
        """
        Handle date/time input programmatically.
        
        Args:
            selector: CSS selector for the date/time input
            value: Value to set (ISO format for dates, e.g., "2024-12-15")
            input_type: Type of input (date, time, datetime-local, etc.)
        """
        # Step 1: Check if native date input
        is_native = self.page.evaluate("""
            (sel) => {
                const el = document.querySelector(sel);
                if (!el) return false;
                return el.tagName.toLowerCase() === 'input' && 
                       ['date', 'time', 'datetime-local', 'month', 'week'].includes(el.type);
            }
        """, selector)
        
        if is_native:
            # Set value directly and dispatch events
            result = self.page.evaluate("""
                (args) => {
                    const el = document.querySelector(args.selector);
                    if (!el) return false;
                    
                    el.value = args.value;
                    el.dispatchEvent(new Event('input', { bubbles: true }));
                    el.dispatchEvent(new Event('change', { bubbles: true }));
                    el.dispatchEvent(new Event('blur', { bubbles: true }));
                    
                    return { value: el.value, type: el.type };
                }
            """, {"selector": selector, "value": value})
            
            print(f"  ✅ Set datetime value: {value}")
            return {"success": True, "result": result, "method": "native"}
        
        # For custom date pickers, try filling as text
        try:
            self.page.fill(selector, value)
            self.page.keyboard.press("Enter")
            print(f"  ✅ Filled custom datetime: {value}")
            return {"success": True, "method": "fill"}
        except Exception as e:
            print(f"  ❌ Datetime handling failed: {e}")
            return {"success": False, "error": str(e)}


# ==================== TESTS ====================

FIXTURE_DIR = Path(__file__).parent
SELECT_FIXTURE = FIXTURE_DIR / "select_fixtures.html"


@pytest.fixture(scope="module")
def page():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 800})
        yield page
        browser.close()


@pytest.fixture
def handler(page):
    return SmartFieldHandler(page)


def test_native_select_programmatic(page, handler):
    """Test native select by programmatically setting value"""
    page.goto(SELECT_FIXTURE.as_uri())
    
    # Select without specifying option - should pick first real option (skip placeholder)
    result = handler.handle_select("#placeholder-select", option_text=None)
    assert result['success']
    
    value = page.eval_on_selector("#placeholder-select", "el => el.value")
    # Should skip "Select one..." placeholder and pick first real option
    assert value in ("coffee", "tea", "juice")
    print(f"  Auto-selected: {value}")


def test_native_select_by_text(page, handler):
    """Test selecting option by text match"""
    page.goto(SELECT_FIXTURE.as_uri())
    
    result = handler.handle_select("#basic-select", option_text="Cherry")
    assert result['success']
    
    value = page.eval_on_selector("#basic-select", "el => el.value")
    assert value == "cherry"


def test_native_select_by_value(page, handler):
    """Test selecting option by value"""
    page.goto(SELECT_FIXTURE.as_uri())
    
    result = handler.handle_select("#optgroup-select", option_value="pasta")
    assert result['success']
    
    value = page.eval_on_selector("#optgroup-select", "el => el.value")
    assert value == "pasta"


def test_custom_dropdown(page, handler):
    """Test custom dropdown with role=option"""
    page.goto(SELECT_FIXTURE.as_uri())
    
    result = handler.handle_select("#custom-dropdown-trigger", option_text="Blue")
    assert result['success']
    
    text = page.text_content("#custom-dropdown-trigger").strip()
    assert text == "Blue"


def test_upload_direct(page, handler):
    """Test file upload by direct input setting"""
    # Create a test fixture HTML with upload
    upload_html = """
    <!DOCTYPE html>
    <html>
    <head><title>Upload Test</title></head>
    <body>
        <label for="file1">Upload File</label>
        <input type="file" id="file1" name="document">
    </body>
    </html>
    """
    page.set_content(upload_html)
    
    # Create a temporary test file
    test_file = FIXTURE_DIR / "test_upload.txt"
    test_file.write_text("test content")
    
    try:
        result = handler.handle_upload("#file1", str(test_file))
        assert result['success']
        
        # Verify file was set
        has_file = page.evaluate("document.querySelector('#file1').files.length > 0")
        assert has_file
    finally:
        test_file.unlink()


def test_upload_with_custom_trigger(page, handler):
    """Test file upload with custom button trigger"""
    upload_html = """
    <!DOCTYPE html>
    <html>
    <head><title>Upload Test</title></head>
    <body>
        <button id="upload-btn" onclick="document.getElementById('hidden-input').click()">
            Choose File
        </button>
        <input type="file" id="hidden-input" style="display: none">
    </body>
    </html>
    """
    page.set_content(upload_html)
    
    test_file = FIXTURE_DIR / "test_upload2.txt"
    test_file.write_text("test content 2")
    
    try:
        result = handler.handle_upload("#upload-btn", str(test_file))
        assert result['success']
        
        has_file = page.evaluate("document.querySelector('#hidden-input').files.length > 0")
        assert has_file
    finally:
        test_file.unlink()


def test_native_date_input(page, handler):
    """Test native date input"""
    date_html = """
    <!DOCTYPE html>
    <html>
    <head><title>Date Test</title></head>
    <body>
        <label for="birthday">Birthday</label>
        <input type="date" id="birthday" name="birthday">
        
        <label for="appointment">Appointment</label>
        <input type="datetime-local" id="appointment" name="appointment">
    </body>
    </html>
    """
    page.set_content(date_html)
    
    # Test date input
    result = handler.handle_datetime("#birthday", "2024-12-25", "date")
    assert result['success']
    
    value = page.eval_on_selector("#birthday", "el => el.value")
    assert value == "2024-12-25"
    
    # Test datetime-local input
    result = handler.handle_datetime("#appointment", "2024-12-25T14:30", "datetime-local")
    assert result['success']
    
    value = page.eval_on_selector("#appointment", "el => el.value")
    assert value == "2024-12-25T14:30"


def test_custom_date_picker(page, handler):
    """Test custom date picker (text input based)"""
    date_html = """
    <!DOCTYPE html>
    <html>
    <head><title>Custom Date Test</title></head>
    <body>
        <label for="custom-date">Select Date</label>
        <input type="text" id="custom-date" placeholder="MM/DD/YYYY">
    </body>
    </html>
    """
    page.set_content(date_html)
    
    result = handler.handle_datetime("#custom-date", "12/25/2024", "text")
    assert result['success']
    
    value = page.eval_on_selector("#custom-date", "el => el.value")
    assert value == "12/25/2024"


def test_select_all_options_accessible(page, handler):
    """Verify we can access options even when select is scrolled out of view"""
    page.goto(SELECT_FIXTURE.as_uri())
    
    # Scroll select out of view
    page.evaluate("window.scrollTo(0, 9999)")
    time.sleep(0.1)
    
    # Should still be able to get options and select
    result = handler.handle_select("#long-label-select", option_text="Production")
    assert result['success']
    
    value = page.eval_on_selector("#long-label-select", "el => el.value")
    assert value == "prod"




