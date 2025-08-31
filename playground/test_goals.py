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

from vision_bot_simplified import VisionBot


def _run_server():
    """Start the Flask server in a background process"""
    from app import create_app
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
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(
            viewport={"width": 1280, "height": 800},
            device_scale_factor=1
        )
        page = context.new_page()
        
        # Initialize the simplified VisionBot
        bot = VisionBot(
            page=page,
            model_name="gemini-2.5-flash-lite",
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
                "goal": "Set 'Location' field to 'London'",
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
                success = bot.achieve_goal(test['goal'])
                
                if success:
                    print(f"✅ SUCCESS: {test['name']}")
                else:
                    print(f"❌ FAILED: {test['name']}")
                
                # Brief pause between tests
                time.sleep(1)
                
            except Exception as e:
                print(f"💥 ERROR in {test['name']}: {e}")
                continue
        
        print(f"\n🎉 Test suite completed!")
        context.close()
        browser.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run VisionBot tests on the playground app")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode")
    parser.add_argument("--only", help="Run specific tests by index or range (e.g., '3' or '1-3' or '1,3,5')")
    parser.add_argument("--match", help="Run tests matching text in name, goal, or URL (case-insensitive)")
    args = parser.parse_args()
    
    # Start Flask server in background
    print("🌐 Starting Flask server...")
    server = multiprocessing.Process(target=_run_server, daemon=True)
    server.start()
    time.sleep(2.0)  # Wait for server to start
    
    try:
        run_tests(headless=args.headless, only=args.only, match=args.match)
    finally:
        print("🛑 Stopping Flask server...")
        server.terminate()
        server.join(timeout=5)
