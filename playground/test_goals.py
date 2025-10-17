# playground/test_goals.py
"""
Simple test suite for the VisionBot using the playground app.
"""
from __future__ import annotations
import time
import multiprocessing
import tempfile
from pathlib import Path
from playwright.sync_api import sync_playwright

# Add project root to path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from vision_bot_refactored import BrowserVisionBot


def _run_server():
    """Start the Flask server in a background process"""
from playground.app import create_app
    app = create_app()
    app.run(host="127.0.0.1", port=5001, debug=False, use_reloader=False)


def _create_test_resume(tmpdir: Path) -> Path:
    """Create a simple test PDF resume"""
    pdf = tmpdir / "resume.pdf"
    if not pdf.exists():
        pdf.write_bytes(
            b"%PDF-1.1\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            b"2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n"
            b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]/Contents 4 0 R>>endobj\n"
            b"4 0 obj<</Length 35>>stream\nBT /F1 12 Tf 72 100 Td (Hello Resume) Tj ET\nendstream endobj\n"
            b"xref\n0 5\n0000000000 65535 f \n0000000010 00000 n \n0000000060 00000 n \n0000000113 00000 n \n0000000207 00000 n \n"
            b"trailer <</Size 5/Root 1 0 R>>\nstartxref\n300\n%%EOF"
        )
    return pdf


def run_tests(base_url: str = "http://127.0.0.1:5001", headless: bool = False, only: str = None, match: str = None):
    """Run the test suite using VisionBot"""
    
    # Start the Flask playground server in a background process
    server_proc = multiprocessing.Process(target=_run_server, daemon=True)
    server_proc.start()
    
    # Brief wait for server to boot
    import socket, time as _t
    for _ in range(50):
        try:
            with socket.create_connection(("127.0.0.1", 5001), timeout=0.2):
                break
        except OSError:
            _t.sleep(0.1)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(
            viewport={"width": 1280, "height": 800},
            device_scale_factor=1
        )
        page = context.new_page()
        
        # Initialize the simplified VisionBot
        bot = BrowserVisionBot(
            page=page,
            max_attempts=5
        )
        
        # Create test resume
        tmpdir = Path(tempfile.gettempdir())
        resume_path = _create_test_resume(tmpdir)
        
        # Define test cases
        tests = [
            {
                "name": "Click Pricing Button",
                "url": f"{base_url}/",
                "goal": "Click the 'Pricing' button",
                "description": "Test basic element clicking"
            },
            {
                "name": "Close Cookie Banner",
                "url": f"{base_url}/modal",
                "goal": "Close the cookie banner",
                "description": "Test modal interaction"
            },
            {
                "name": "Load Infinite Results",
                "url": f"{base_url}/infinite?total=100&chunk=20",
                "goal": "Load at least 80 results by scrolling",
                "description": "Test infinite scrolling"
            },
            {
                "name": "Open List Item",
                "url": f"{base_url}/list?n=30&prefix=Story",
                "goal": "Open the first item in the list",
                "description": "Test list item interaction"
            },
            {
                "name": "Apply Filters",
                "url": f"{base_url}/filters",
                "goal": "Filter by 'Remote' and 'iOS'",
                "description": "Test filter application"
            },
            {
                "name": "Fill Form Field",
                "url": f"{base_url}/form",
                "goal": "Fill the form field",
                "description": "Test form field filling"
            },
            {
                "name": "Complete Form",
                "url": f"{base_url}/form",
                "goal": "Fill all required fields and submit the application",
                "description": "Test complete form submission"
            },
            {
                "name": "Upload Resume",
                "url": f"{base_url}/upload",
                "goal": f"Upload the resume file from {resume_path}",
                "description": "Test file upload functionality"
            },
            {
                "name": "Login",
                "url": f"{base_url}/login",
                "goal": "Log in with username 'standard_user' and password 'secret_sauce'",
                "description": "Test login functionality"
            },
            {
                "name": "Open New Tab",
                "url": f"{base_url}/newtab",
                "goal": "Open 'Docs' in a new tab",
                "description": "Test new tab functionality"
            },
            {
                "name": "Test Traditional Select",
                "url": f"{base_url}/form",
                "goal": "Select an option from the Location dropdown",
                "description": "Test traditional HTML select field handling"
            },
            {
                "name": "Test Custom Select",
                "url": f"{base_url}/form",
                "goal": "Choose an option from the custom dropdown field",
                "description": "Test custom select field with AI option selection"
            },
            {
                "name": "Test File Upload",
                "url": f"{base_url}/upload",
                "goal": "Upload a file using the file upload button",
                "description": "Test file upload field with user interaction pause"
            },
            {
                "name": "Test Date Field",
                "url": f"{base_url}/form",
                "goal": "Set a date in the date picker field",
                "description": "Test date input field handling"
            },
            {
                "name": "Test DateTime Field",
                "url": f"{base_url}/form",
                "goal": "Set a date and time in the datetime field",
                "description": "Test datetime-local input field handling"
            },
            {
                "name": "Test Custom Select Experience",
                "url": f"{base_url}/form",
                "goal": "Select 'Senior (5+ years)' from the choose experience level field",
                "description": "Test custom select field with AI option selection"
            },
            {
                "name": "Test Date Field Start Date",
                "url": f"{base_url}/form",
                "goal": "Set the start date to 1 September, 2025",
                "description": "Test HTML5 date input field"
            },
            {
                "name": "Test Time Field Interview",
                "url": f"{base_url}/form",
                "goal": "Set the interview time to 2:30 PM",
                "description": "Test HTML5 time input field"
            },
            {
                "name": "Test DateTime Appointment",
                "url": f"{base_url}/form",
                "goal": "set only the appointment schedule for January 20, 2025 at 3:45 PM",
                "description": "Test HTML5 datetime-local input field"
            }
        ]
        
        # Filter tests based on --only and --match flags
        if only:
            filtered_tests = []
            for token in only.split(","):
                token = token.strip()
                if "-" in token:
                    # Handle ranges like "1-3"
                    try:
                        start, end = map(int, token.split("-"))
                        start, end = min(start, end), max(start, end)
                        filtered_tests.extend(tests[start-1:end])
                    except ValueError:
                        print(f"⚠️ Invalid range: {token}")
                else:
                    # Handle single index
                    try:
                        idx = int(token) - 1  # Convert to 0-based
                        if 0 <= idx < len(tests):
                            filtered_tests.append(tests[idx])
                        else:
                            print(f"⚠️ Test index {token} out of range (1-{len(tests)})")
                    except ValueError:
                        print(f"⚠️ Invalid test index: {token}")
            
            if filtered_tests:
                tests = filtered_tests
                print(f"🔍 Running {len(tests)} selected tests: {only}")
            else:
                print("❌ No valid tests selected with --only flag")
                return
        
        if match:
            match_lower = match.lower()
            tests = [t for t in tests if 
                    match_lower in t['name'].lower() or 
                    match_lower in t['goal'].lower() or 
                    match_lower in t['url'].lower()]
            
            if tests:
                print(f"🔍 Running {len(tests)} tests matching '{match}'")
            else:
                print(f"❌ No tests match '{match}'")
                return
        
        print(f"🚀 Running {len(tests)} tests with VisionBot...")
        
        for i, test in enumerate(tests, 1):
            print(f"\n{'='*80}")
            print(f"TEST {i}/{len(tests)}: {test['name']}")
            print(f"URL: {test['url']}")
            print(f"Goal: {test['goal']}")
            print(f"Description: {test['description']}")
            print(f"{'='*80}")
            
            try:
                # Navigate to test page
                page.goto(test['url'], wait_until="domcontentloaded")
                time.sleep(0.5)  # Let page settle
                
                # Attempt to achieve the goal
                print(f"🎯 Attempting: {test['goal']}")
                success = bot.act(test['goal'])
                
                if success:
                    print(f"✅ SUCCESS: {test['name']}")
                else:
                    print(f"❌ FAILED: {test['name']}")
                
                # Brief pause between tests
                time.sleep(1)
                
            except Exception as e:
                print(f"💥 ERROR in {test['name']}: {e}")
                continue
        
        print("\n🎉 Test suite completed!")
        context.close()
        browser.close()
        try:
            server_proc.terminate()
        except Exception:
            pass


def test_new_action_types(base_url: str = "http://127.0.0.1:5001", headless: bool = False):
    """Dedicated test for the new action types: HANDLE_SELECT, HANDLE_UPLOAD, HANDLE_DATETIME"""
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(
            viewport={"width": 1280, "height": 800},
            device_scale_factor=1
        )
        page = context.new_page()
        
        # Initialize the VisionBot
        bot = BrowserVisionBot(
            page=page,
            max_attempts=3
        )
        
        # Test cases specifically for new action types
        action_tests = [
            {
                "name": "Traditional HTML Select",
                "url": f"{base_url}/form",
                "goal": "Select 'London' from the Location dropdown",
                "description": "Tests HANDLE_SELECT with traditional <select> element",
                "expected_action": "HANDLE_SELECT",
                "field_type": "traditional_select"
            },
            {
                "name": "Custom Dropdown Select",
                "url": f"{base_url}/form",
                "goal": "Choose an experience level from the custom dropdown",
                "description": "Tests HANDLE_SELECT with custom implementation + AI",
                "expected_action": "HANDLE_SELECT",
                "field_type": "custom_select"
            },
            {
                "name": "File Upload Field",
                "url": f"{base_url}/upload",
                "goal": "Upload a file using the upload button",
                "description": "Tests HANDLE_UPLOAD with user interaction pause",
                "expected_action": "HANDLE_UPLOAD",
                "field_type": "file_upload"
            },
            {
                "name": "Date Input Field",
                "url": f"{base_url}/form",
                "goal": "Set the start date to December 15, 2024",
                "description": "Tests HANDLE_DATETIME with date input",
                "expected_action": "HANDLE_DATETIME",
                "field_type": "date_input"
            },
            {
                "name": "DateTime Input Field",
                "url": f"{base_url}/form",
                "goal": "Set appointment datetime to December 15, 2024 at 2:30 PM",
                "description": "Tests HANDLE_DATETIME with datetime-local input",
                "expected_action": "HANDLE_DATETIME",
                "field_type": "datetime_input"
            },
            {
                "name": "Custom Select Experience Level",
                "url": f"{base_url}/form",
                "goal": "Select 'Senior (5+ years)' from the Experience Level dropdown",
                "description": "Tests HANDLE_SELECT with custom dropdown + AI selection",
                "expected_action": "HANDLE_SELECT",
                "field_type": "custom_select"
            },
            {
                "name": "Start Date Field",
                "url": f"{base_url}/form",
                "goal": "Set the start date to March 15, 2025",
                "description": "Tests HANDLE_DATETIME with HTML5 date input",
                "expected_action": "HANDLE_DATETIME",
                "field_type": "date_input"
            },
            {
                "name": "Interview Time Field",
                "url": f"{base_url}/form",
                "goal": "Set the interview time to 10:30 AM",
                "description": "Tests HANDLE_DATETIME with HTML5 time input",
                "expected_action": "HANDLE_DATETIME",
                "field_type": "time_input"
            },
            {
                "name": "Appointment DateTime Field",
                "url": f"{base_url}/form",
                "goal": "Schedule appointment for March 20, 2025 at 2:15 PM",
                "description": "Tests HANDLE_DATETIME with datetime-local input",
                "expected_action": "HANDLE_DATETIME",
                "field_type": "datetime_local"
            }
        ]
        
        print("🧪 Testing New Action Types: HANDLE_SELECT, HANDLE_UPLOAD, HANDLE_DATETIME")
        print(f"{'='*80}")
        
        for i, test in enumerate(action_tests, 1):
            print(f"\n🎯 ACTION TEST {i}/{len(action_tests)}: {test['name']}")
            print(f"URL: {test['url']}")
            print(f"Goal: {test['goal']}")
            print(f"Expected Action Type: {test['expected_action']}")
            print(f"Field Type: {test['field_type']}")
            print(f"{'='*60}")
            
            try:
                # Navigate to test page
                page.goto(test['url'], wait_until="domcontentloaded")
                time.sleep(1)  # Let page settle
                
                print(f"📋 Attempting goal: {test['goal']}")
                
                # Special instructions for file upload test
                if test['field_type'] == 'file_upload':
                    print("📁 NOTE: This test will pause for manual file selection.")
                    print("    When prompted, please select any file from your system.")
                    print("    The test will continue after you select a file and press Enter.")
                
                # Attempt to achieve the goal
                success = bot.act(test['goal'])
                
                if success:
                    print(f"✅ SUCCESS: {test['name']}")
                    print(f"    The {test['expected_action']} action type worked correctly!")
                else:
                    print(f"❌ FAILED: {test['name']}")
                    print(f"    The {test['expected_action']} action may need adjustment.")
                
                # Brief pause between tests
                time.sleep(2)
                
            except Exception as e:
                print(f"💥 ERROR in {test['name']}: {e}")
                continue
        
        print("\n🎉 New Action Types Test Suite Completed!")
        print("📊 Summary:")
        print("   • HANDLE_SELECT: Handles both traditional and custom select fields")
        print("   • HANDLE_UPLOAD: Manages file uploads with user interaction")
        print("   • HANDLE_DATETIME: Processes date and datetime input fields")
        
        context.close()
        browser.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run VisionBot tests on the playground app")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode")
    parser.add_argument("--only", help="Run specific tests by index or range (e.g., '3' or '1-3' or '1,3,5')")
    parser.add_argument("--match", help="Run tests matching text in name, goal, or URL (case-insensitive)")
    parser.add_argument("--action-types", action="store_true", help="Run only the new action types tests (HANDLE_SELECT, HANDLE_UPLOAD, HANDLE_DATETIME)")
    args = parser.parse_args()
    
    # Start Flask server in background
    print("🌐 Starting Flask server...")
    server = multiprocessing.Process(target=_run_server, daemon=True)
    server.start()
    time.sleep(2.0)  # Wait for server to start
    
    try:
        if args.action_types:
            test_new_action_types(headless=args.headless)
        else:
            run_tests(headless=args.headless, only=args.only, match=args.match)
    finally:
        print("🛑 Stopping Flask server...")
        server.terminate()
        server.join(timeout=5)
