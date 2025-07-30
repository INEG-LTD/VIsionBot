#!/usr/bin/env python3
"""
Robust Test Application Filler

A more robust test script that handles network issues and provides better debugging
"""

import time
from hybrid_browser_bot import HybridBrowserBot
from application_filler import ApplicationFiller

    
# Test preferences for iOS Engineer position
preferences = {
    'name': 'John Doe',
    'resume_path': 'cv.pdf',
    'cover_letter_path': 'cover_letter.pdf',
    'photo_path': 'photo.png',
    'email': 'john.doe@example.com',
    'phone': '+1234567890',
    'address': '123 Main St, Anytown, USA',
    'city': 'Bristol',
    'country': 'United Kingdom',
    'county': 'Gloucestershire',
    'state': 'England',
    'postcode': 'BS1 1AA',
    'age': 30,
    'gender': 'Male',
    'date_of_birth': '1990-01-01',
    'nationality': 'British',
    'marital_status': 'Single',
    'driving_license': 'Yes',
    'car_ownership': 'Yes',
    'car_model': 'Toyota Corolla',
    'car_year': 2020,
    'disability': 'No',
    'disability_description': 'No',
    'disability_type': 'No',
    'disability_description': 'No',
    'religion': 'Christian',
    'relationship_status': 'Single',
    'relationship_status_description': 'No',
    'relationship_status_type': 'No',
    'relationship_status_description': 'No',
    'ethnicity': 'White',
    'ethnicity_description': 'No',
    'ethnicity_type': 'No',
    'ethnicity_description': 'No',
    'pregnant': 'No',
    'zip': '12345',
    'job_titles': ['iOS Engineer', 'Senior iOS Engineer', 'Mobile Developer'],
    'locations': ['London', 'UK', 'United Kingdom'],
    'salary_min': 80000,
    'employment_types': ['Full-time'],
    'required_skills': ['Swift', 'iOS', 'SwiftUI', 'Combine', 'Mobile Development'],
    'experience_levels': ['Senior', 'Mid-level'],
    'visa_sponsorship_required': False,
    'remote_flexibility': ['Hybrid', 'Remote'],
    'desired_benefits': ['Health Insurance', 'Stock Options', 'Learning Budget'],
    'exclude_keywords': ['unpaid', 'internship', 'junior']
}

def test_application_filler_robust():
    """Test the application filler with better error handling"""
    
    # SumUp job URL from the user
    test_url = "https://www.sumup.com/careers/positions/london-england-united-kingdom/ios/senior-ios-engineer-global-bank/8048304002/?gh_jid=8048304002&gh_src=jn5gvww32us"
    
    print("🚀 Starting Robust Application Filler Test")
    print("=" * 60)
    print(f"🔗 Test URL: {test_url}")
    print(f"🎯 Job Title: Senior iOS Engineer - Global Bank")
    print(f"🏢 Company: SumUp")
    print(f"📍 Location: London, England, United Kingdom")
    print("=" * 60)
    
    # Initialize browser bot
    bot = HybridBrowserBot(headless=False, preferences=preferences)
    
    try:
        # Start browser and navigate to job page with retries
        print("🌐 Starting browser...")
        if not bot.start_browser():
            print("❌ Failed to start browser")
            return False
        
        # Try navigation with multiple approaches
        navigation_success = False
        
        # Method 1: Direct navigation with shorter timeout
        try:
            print(f"📍 Attempting direct navigation (Method 1)...")
            bot.page.goto(test_url, wait_until='domcontentloaded', timeout=15000)
            time.sleep(3)  # Wait for additional content to load
            navigation_success = True
            print("✅ Direct navigation successful")
        except Exception as e:
            print(f"⚠️ Direct navigation failed: {e}")
        
        # Method 2: Navigate to base domain first, then to specific page
        if not navigation_success:
            try:
                print(f"📍 Attempting base domain navigation (Method 2)...")
                bot.page.goto("https://www.sumup.com", wait_until='domcontentloaded', timeout=10000)
                time.sleep(2)
                bot.page.goto(test_url, wait_until='domcontentloaded', timeout=15000)
                time.sleep(3)
                navigation_success = True
                print("✅ Base domain navigation successful")
            except Exception as e:
                print(f"⚠️ Base domain navigation failed: {e}")
        
        # Method 3: Manual navigation
        if not navigation_success:
            print("📍 Manual navigation required...")
            print(f"Please manually navigate to: {test_url}")
            print("Press 'Resume' when the page is loaded.")
            bot.page.pause()
            navigation_success = True
            print("✅ Manual navigation completed")
        
        if not navigation_success:
            print("❌ All navigation methods failed")
            return False
        
        # Take screenshot of initial page
        try:
            screenshot_path = bot.take_screenshot("sumup_job_page_initial.png")
            print(f"📸 Initial page screenshot: {screenshot_path}")
        except Exception as e:
            print(f"⚠️ Screenshot failed: {e}")
        
        # Wait a bit more for dynamic content
        print("⏳ Waiting for page to fully load...")
        time.sleep(5)
        
        # Check if page loaded properly
        try:
            page_title = bot.page.title()
            current_url = bot.page.url
            print(f"📄 Page title: {page_title}")
            print(f"🔗 Current URL: {current_url}")
            
            # Check if we're on the right page
            if "sumup" not in current_url.lower() or "careers" not in current_url.lower():
                print("⚠️ May not be on the correct job page")
        except Exception as e:
            print(f"⚠️ Error checking page info: {e}")
        
        # Initialize application filler
        print("🤖 Initializing Application Filler...")
        app_filler = ApplicationFiller(bot.page, preferences)
        
        # Test individual components first
        print("\n🧪 Testing Individual Components...")
        print("=" * 40)
        
        # Test 1: Apply button detection
        # try:
        #     print("🔍 Testing apply button detection...")
        #     apply_result = app_filler.check_and_click_apply_button()
        #     print(f"Apply button result: {apply_result}")
        # except Exception as e:
        #     print(f"❌ Apply button test failed: {e}")
        
        
        # Test 2: Form field detection
        # try:
        #     print("🔍 Testing form field detection...")
        #     form_fields = app_filler.find_all_form_inputs()
        #     print(f"Form fields: {form_fields}")
        # except Exception as e:
        #     print(f"❌ Form field test failed: {e}")
        #     form_fields = {'total_fields': 0}
        
        # Decide whether to run full algorithm
        if True:
            print("\n🚀 Running Full Application Filling Process...")
            print("=" * 50)
            
            try:
                success = app_filler.fill_application()
                
                if success:
                    print("\n🎉 Application filling completed successfully!")
                    try:
                        bot.take_screenshot("sumup_application_completed.png")
                    except:
                        pass
                else:
                    print("\n⚠️ Application filling encountered issues")
                    try:
                        bot.take_screenshot("sumup_application_issues.png")
                    except:
                        pass
            except Exception as e:
                print(f"\n❌ Application filling error: {e}")
                try:
                    bot.take_screenshot("sumup_application_error.png")
                except:
                    pass
        else:
            print("\n📝 No form fields detected - this may be a job description page")
            print("You may need to click an 'Apply' button first to access the application form.")
            
            # Try to find and show any buttons on the page
            try:
                buttons = bot.page.query_selector_all('button, a[href*="apply"], [class*="apply"]')
                if buttons:
                    print(f"🔍 Found {len(buttons)} potential buttons/links:")
                    for i, button in enumerate(buttons[:5]):  # Show first 5
                        try:
                            text = button.inner_text().strip()[:50]
                            if text:
                                print(f"  {i+1}. '{text}'")
                        except:
                            pass
            except Exception as e:
                print(f"⚠️ Error finding buttons: {e}")
        
        # Keep browser open for manual inspection
        print("\n⏸️ Test complete. Browser will remain open for inspection.")
        print("You can manually review the page and application process.")
        print("Click 'Resume' when you're done to close the browser.")
        bot.page.pause()
        
        return True
        
    except Exception as e:
        print(f"❌ Error in test: {e}")
        try:
            bot.take_screenshot("test_error.png")
        except:
            pass
        return False
    
    finally:
        # Clean up
        try:
            bot.stop_browser()
            print("✅ Browser closed")
        except:
            pass


def test_simple_application_form():
    """Test with a simple form page that should work more reliably"""
    
    print("🧪 Running Simple Application Form Test")
    print("=" * 40)
    print("This test will open a browser and allow you to navigate")
    print("to any job application form to test the ApplicationFiller.")
    
    bot = HybridBrowserBot(headless=False, preferences=preferences)
    
    try:
        if not bot.start_browser():
            return False
        
        # Start with a blank page
        bot.page.goto("about:blank")
        
        print("\n📝 Manual Test Instructions:")
        print("1. Navigate to any job application page")
        print("2. Click 'Resume' when you're ready to test the ApplicationFiller")
        print("3. The ApplicationFiller will try to fill the form automatically")
        
        bot.page.pause()
        
        # Test the application filler on whatever page the user navigated to
        app_filler = ApplicationFiller(bot.page, preferences)
        
        print("🤖 Running ApplicationFiller on current page...")
        success = app_filler.fill_application()
        
        if success:
            print("🎉 ApplicationFiller completed!")
        else:
            print("⚠️ ApplicationFiller encountered issues")
        
        print("\n⏸️ Test complete. Click 'Resume' to close the browser.")
        bot.page.pause()
        
        return success
        
    except Exception as e:
        print(f"❌ Error in simple test: {e}")
        return False
    
    finally:
        try:
            bot.stop_browser()
        except:
            pass


if __name__ == "__main__":
    print("🎮 Robust Application Filler Test Suite")
    print("Choose test mode:")
    print("1. SumUp Job Test (with robust navigation)")
    print("2. Manual Job Application Test (you choose the site)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        test_simple_application_form()
    else:
        test_application_filler_robust() 