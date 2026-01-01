#!/usr/bin/env python3
"""
Integration test to reproduce and verify the typing fix for long text.
This test creates a simple HTML page with a textarea and tests typing long text.
"""

import time
import os
import tempfile
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
import sys
sys.path.insert(0, str(project_root))

from browser_vision_bot import BrowserVisionBot

def test_long_text_typing():
    """Test typing long text into a textarea"""

    # Create a simple HTML test page
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Typing Test</title>
    </head>
    <body>
        <h1>Typing Test</h1>
        <textarea id="test-textarea" placeholder="Type here..." rows="10" cols="50"></textarea>
        <br><br>
        <button id="submit-btn">Submit</button>
    </body>
    </html>
    """

    # Create temporary HTML file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
        f.write(html_content)
        temp_html_path = f.name

    try:
        # Initialize bot
        bot = BrowserVisionBot()

        # Start the bot
        bot.start()

        # Navigate to the test page
        test_url = f"file://{temp_html_path}"
        bot.page.goto(test_url)

        # Wait for page to load
        time.sleep(1)

        # Test text - same length as the ElevenLabs example
        long_text = "I am excited by ElevenLabs' mission to advance voice AI and make high-quality, natural-sounding speech accessible to creators and users. I would love to apply my iOS experience to help bring these capabilities to mobile platforms in an intuitive, reliable app."

        print(f"Testing typing of long text ({len(long_text)} characters):")
        print(f"Text: {long_text[:100]}...")

        print(f"About to call bot.act() with text length: {len(long_text)}")

        # Use act() to type into the textarea
        result = bot.act(f"type: {long_text} in textarea")

        if result.success:
            print("✅ Typing action succeeded")

            # First, let's try to get the textarea value using JavaScript directly
            try:
                textarea_value = bot.page.evaluate("document.getElementById('test-textarea').value")
                print(f"Direct JavaScript extraction ({len(textarea_value)} chars): {textarea_value[:100]}...")
            except Exception as e:
                print(f"JavaScript extraction failed: {e}")

            # Verify what was actually typed by extracting the textarea value
            extract_result = bot.extract("Get the text from the textarea", output_format="text")
            if extract_result.success:
                typed_text = extract_result.data.strip()
                print(f"Extracted text ({len(typed_text)} chars): {typed_text[:100]}...")

                # Normalize whitespace for comparison (extract may add line breaks)
                normalized_typed = ' '.join(typed_text.split())
                normalized_expected = ' '.join(long_text.split())

                if normalized_typed == normalized_expected:
                    print("✅ SUCCESS: Full text was typed correctly!")
                    return True
                else:
                    print("❌ FAILURE: Text was truncated or modified")
                    print(f"Expected length: {len(long_text)}")
                    print(f"Got length: {len(typed_text)}")
                    print(f"Expected (normalized): {normalized_expected}")
                    print(f"Got (normalized): {normalized_typed}")
                    return False
            else:
                print("❌ Could not extract text from textarea")
                return False
        else:
            print(f"❌ Typing action failed: {result.message}")
            return False

    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Clean up
        try:
            bot.stop()
        except:
            pass

        try:
            os.unlink(temp_html_path)
        except:
            pass

if __name__ == "__main__":
    print("Running typing integration test...")
    success = test_long_text_typing()
    if success:
        print("\n🎉 Test PASSED - typing works correctly!")
    else:
        print("\n💥 Test FAILED - typing issue still exists!")
    sys.exit(0 if success else 1)
